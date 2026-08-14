"""The baseline ReAct agent — deliberately thin, deliberately weak.

STUDENT-OWNED. This is the agent you start from. It runs the loop, it
routes every model call and every tool call through your middleware, and
it writes a conforming trace. What it does NOT do is any of the five
jobs the layers exist for: it never checks a citation, never notices a
fabrication, never resists an injected instruction, never respects the
tool budget, never retries a broken tool call. On the trap-spanning brief
set it scores ~38 of 100 and it fails visibly, which is the point — every
point above that is a layer you built.

WHAT YOU GET FOR FREE
=====================

**The trace gate passes out of the box.** `run()` emits `agent_start`,
one `model_call` per turn (with the tokens and the model's raw output
text the scorer needs), and `agent_end`; `arena/tools.py` emits its own
`tool_call` events. Keep using the harness and `Trace.validate` says
`(True, "")` without you doing anything. Bypass it — call the model
directly, hand-write JSONL — and the gate fails, which zeroes the entry.
The gate is PASS/FAIL, never a scored dimension.

**Your claims keep their provenance.** The report is extracted with
`arena.model.parse_output`, the same frozen parser the scorer credits
through, applied to the same canonicalised text. Do not swap in a
friendlier parser of your own: a lenient one happily builds a
plausible-looking report out of text the scorer will not recognise as a
FINAL, and then EVERY claim scores `NOT_FROM_MODEL`. Measured cost of
that mistake: a silent 40.15 instead of 92.52 — a run that looks perfect
and scores like a troll.

THE LOOP, IN ORDER
==================

    before_agent
    repeat up to MAX_STEPS times:
        messages_out = before_model(history)
        response     = wrap_model_call(model.complete)(messages_out)
        emit model_call(prompt_tokens, completion_tokens, output_text)
        response     = after_model(response)
        parsed       = parse_output(canonicalise(response.text))
        if parsed is a FINAL:  break
        result       = wrap_tool_call(dispatch)(tool, args)
        history     += [assistant(response.text), user(observation)]
    report = after_agent(parsed.final or {})
    tools.submit(report)
    emit agent_end

`MAX_STEPS` is 40 and must not be lowered. Under a fully hostile tool
layer the mock needs 31 model turns to reach a FINAL; a cap below that
produces no report at all, silently, and only on the unlucky seeds.

TWO THINGS THIS AGENT DOES ON PURPOSE, AND WHY
==============================================

1. `before_model` is applied to a COPY of the history, and only the raw
   response and the raw observation are appended back. So a layer that
   appends a one-turn nudge (`budget_policy`) nudges for one turn instead
   of forever.
2. `tools.submit()` is called directly, NOT through `wrap_tool_call`.
   Submitting is the run's own bookkeeping rather than an action the
   agent chose, and a `retry` layer that re-submitted would spend budget
   the scorer counts (`tools.calls` includes `submit`) for nothing: a
   timed-out submit still records the report verbatim on the trace.

THE SYSTEM PROMPT THIS AGENT SENDS
==================================

`ARENA_SYSTEM_PROMPT` is frozen in `arena/model.py` and was written for
`MockModel`, which is templated to always act. A real endpoint is not,
and the difference was measured on live keys:

    gpt-5.6-luna abstained on TURN 1 with ZERO tool calls on 4 of 6 runs
    (contradiction 2/2, refund 2/2). Zero tools -> zero claims -> the
    abstain floor -> a ladder with no gradient. deepseek-v4-flash: 0/6.

So this module ships `REAL_MODEL_PROMPT_ADDENDUM` and the prompt that
carries it, `ARENA_SYSTEM_PROMPT_REAL`. Nothing in `arena/` is unfrozen:
the addendum is appended by student-owned code and handed to the agent
through the keyword argument that already existed.

    ReActAgent(model, tools, trace, system_prompt=ARENA_SYSTEM_PROMPT_REAL)

**THE SCORED, REAL-MODEL PATH MUST CONSTRUCT THE AGENT THAT WAY.**

The DEFAULT is still the bare frozen `ARENA_SYSTEM_PROMPT`, and that is a
measured decision rather than caution. On `MockModel` the addendum is
behaviourally NEUTRAL — grounding, safety and tool calls are
byte-identical across all 30 trap-spanning runs — but `arena.model`
estimates prompt tokens as `len(conversation) // 4`, so a 2,792-character
addendum adds ~698 tokens to EVERY turn of a mock run and costs 1.28
points of efficiency against the mock's 12,000-token budget (14.39 ->
13.11), moving the practice ladder from 92.52 to 91.24. That is an
artefact of the mock's estimator, not a real cost, and the practice
ladder is a fixed acceptance artefact. Defaulting it off keeps the two
paths honest: the mock ladder stays byte-identical, and the real path
opts in explicitly.

The ~700 prompt tokens per call ARE a real cost on a real endpoint, and
the scored round's per-brief `max_tokens` is sized with them included. If
you switch the addendum on, measure your own efficiency delta with
`scripts/run_practice.py --prompt-addendum` before assuming it is free.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field

from arena.model import (
    ARENA_SYSTEM_PROMPT,
    TOOL_ERROR_PREFIX,
    parse_output,
)
from arena.tools import ToolResult

from harness.middleware import Middleware, MiddlewareStack

#: Hard ceiling on model turns. >= 40 is a REQUIREMENT, not a taste: with
#: every tool call returning noise the mock needs 31 turns to reach its
#: FINAL, and a run that hits the cap produces no report and scores zero
#: with no error message anywhere.
MAX_STEPS = 40

#: `k` a search is allowed to ask for. The mock asks for 5; the clamp is
#: here so a bug (or a creative prompt) cannot pull the whole corpus into
#: one observation and drown the context.
MAX_SEARCH_K = 20

#: Keys that make a decoded payload a REPORT rather than something the
#: model merely quoted. Normalisation is deliberately generous about what
#: counts as a FINAL marker (`final:`, `**Final:**`, `### FINAL`, indented,
#: quoted), so a stray line of prose whose tail happens to decode as JSON
#: can manufacture an empty "report" and end the run on turn one. A
#: payload carrying none of these keys is not a report.
#:
#: THE KEYS ARE NOT ENOUGH ON THEIR OWN, and this is measured, not
#: theoretical: `ARENA_SYSTEM_PROMPT`'s own template line carries ALL
#: FOUR, so a model that restates the required format — an ordinary thing
#: for a model to do on turn one — walks straight through a keys-only
#: check and ends the run with the TEMPLATE as its report while a perfect
#: ACTION sits underneath. Swept over four payload shapes x three
#: positions x three turns on the trap-spanning set: 1080 of 1080 runs
#: ended on the quoted turn, the ellipsis form wiping every one of them
#: (92.52 -> 0.00). With the content check below: 0 of 1080 for every
#: placeholder shape. Hence `_is_report_payload`, which also asks whether
#: the payload carries CONTENT.
REPORT_KEYS = ("answer", "claims", "abstain", "citations")

#: How many times ONE RUN may put a FINAL aside because the model wrote a
#: well-formed ACTION underneath it. Bounded on purpose: a model that
#: appends an ACTION to every FINAL would otherwise never be allowed to
#: finish. After this many deferrals the FINAL is taken at face value.
MAX_FINAL_DEFERRALS = 2

#: Opt-in switch used by the practice CLI. Reflection costs one additional
#: model call, so it must not silently change the deterministic mock ladder.
SELF_REFLECTION_ENV = "ARENA_SELF_REFLECTION"
SELF_REFLECTION_SENTINEL = "[ARENA-SELF-REFLECTION]"
SELF_REFLECTION_PROMPT = f"""{SELF_REFLECTION_SENTINEL}
Bản FINAL vừa rồi chỉ là bản nháp. Hãy tự phản biện đúng một lần trước khi nộp:
1. Mỗi claim có phải là chuỗi trích nguyên văn nằm gọn trong một dòng tài liệu đã đọc không?
2. Mỗi doc_id có đúng là tài liệu chứa claim đó và đã được fetch không?
3. Nếu bằng chứng thiếu, mâu thuẫn hoặc không hỗ trợ kết luận, hãy đặt abstain thành true.
4. Loại bỏ suy đoán và không thêm dữ kiện mới ngoài các observation đã có.
5. Một claim được tài liệu hỗ trợ nhưng bị cắt quá ngắn vẫn có thể không phủ đủ dữ kiện.
   Với dòng liên quan không quá 400 ký tự, hãy chép TRỌN DÒNG; đặc biệt không cắt mất
   điều kiện, ngoại lệ, phạm vi áp dụng, trạng thái hiệu lực hoặc câu tiếp nối trên cùng dòng.
Không gọi thêm công cụ. Trả về ngay một FINAL JSON đã sửa, đúng giao thức hiện tại."""

REAL_EVIDENCE_PREFETCH_ENV = "ARENA_REAL_EVIDENCE_PREFETCH"
PREFETCH_SENTINEL = "[ARENA-PREFETCHED-EVIDENCE]"
PREFETCH_SEARCH_K = 4
PREFETCH_MAX_DOCS = 3
PREFETCH_MAX_LINES = 4

PREFETCH_PROMPT = """{sentinel}
Harness đã thực hiện search/fetch trước lượt trả lời này và đưa nguyên văn các
dòng bằng chứng dưới đây. Không gọi thêm công cụ. Hãy trả lời ngay bằng một
dòng FINAL JSON duy nhất.

Quy tắc bắt buộc:
- claims chỉ được lấy từ các dòng EVIDENCE bên dưới, chép nguyên văn text và doc_id.
- Chép TOÀN BỘ dòng sau dấu hai chấm của EVIDENCE, không rút gọn, không tách câu,
  không bỏ câu thứ hai trên cùng dòng.
- Nếu có hai nguồn chính thống mâu thuẫn, đưa cả hai dòng vào claims và đặt abstain true.
- Nếu câu hỏi yêu cầu chọn một verdict trong các phương án (a), (b), (c), thêm khóa
  verdict và chọn đúng một câu. Khi bằng chứng chỉ là một thống kê nội bộ, chọn phương
  án nói rằng con số chưa đủ để kết luận.
- Với câu hỏi hỏi phòng ban/thời hạn/tỷ lệ, ưu tiên dòng bắt đầu bằng "Mọi trường hợp".
- Với câu hỏi hỏi con số/thống kê, ưu tiên dòng bắt đầu bằng "Trong kỳ báo cáo".
- Với câu hỏi về ticket, đổi trả, hoàn tiền hoặc "đã được xử lý thế nào", nếu EVIDENCE
  có dòng bắt đầu bằng "Trong kỳ báo cáo" cùng chủ đề, hãy dùng dòng đó làm bằng chứng
  xử lý; không đặt abstain chỉ vì dòng nói đây là báo cáo nội bộ.

EVIDENCE:
{evidence}
"""

_WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)

_STOPWORDS = {
    "a", "an", "anh", "bao", "bên", "bị", "bộ", "các", "cái", "cần", "cấp",
    "cho", "chung", "có", "của", "đã", "đang", "được", "để", "đến", "đi",
    "đó", "hỏi", "khi", "không", "là", "lần", "mà", "mấy", "một", "nào",
    "nêu", "này", "như", "những", "phải", "qua", "ra", "rằng", "sau", "sự",
    "theo", "thế", "thì", "trong", "từ", "và", "về", "vì", "với",
}

#: What a model writes where CONTENT belongs when it is QUOTING the
#: protocol instead of answering: the template's own `...`, an ellipsis,
#: a dash, or an `<angle-bracket slot>`.
_PLACEHOLDER_RE = re.compile(r"\A[\s.…·\-–—]*\Z")

#: The frozen parser's FINAL marker is `^FINAL:` at a line start
#: (`arena.model._FINAL_RE`). Used ONLY to locate marker lines — every
#: payload on this path is still decoded by `parse_output` itself.
_FINAL_MARKER = "FINAL:"

# ---------------------------------------------------------------------------
# The real-model prompt addendum
# ---------------------------------------------------------------------------

#: Appended to `ARENA_SYSTEM_PROMPT` for the scored, real-model path.
#:
#: THREE CLAUSES, EACH ANSWERING A MEASURED FAILURE:
#:
#: A. **Search before abstaining.** gpt-5.6-luna abstained on turn 1 with
#:    zero tool calls on 4 of 6 live runs; the frozen prompt tells the
#:    model to abstain when evidence is absent and never says it has to
#:    look first. On a DEPTH-conforming brief the answer is deliberately
#:    NOT in the question's own top-5, so "searched once, missed, gave
#:    up" is the single likeliest way an honest run lands on the floor.
#:    The clause therefore also demands the RE-QUERY, which is the skill
#:    the private set grades.
#:
#: B. **Strict JSON on the marker's own line.** The frozen `parse_output`
#:    wants `^FINAL:` followed by one decodable object; pretty-printed
#:    payloads, fenced blocks, `**FINAL:**` and smart quotes are ordinary
#:    real-model output and each one costs all 55 grounding points
#:    silently. `arena.scorer._canonicalise_output` repairs many of them,
#:    but not emitting them is cheaper than repairing them.
#:
#: C. **The schema in WORDS, with no quotable template.** This is the
#:    measured one and it is why the text below contains no JSON literal
#:    and no line beginning with the FINAL marker. `ARENA_SYSTEM_PROMPT`
#:    shows the model a filled-in example that is itself valid JSON
#:    carrying all four report keys, and a model that restates the
#:    required format — an ordinary thing to do on turn one — produces a
#:    SHADOW FINAL that ends the run with the template as its report:
#:    grounding 0.00, total 40.15 through the real agent. `_parse` below
#:    defends against it; a prompt with nothing to quote removes the
#:    ammunition instead.
#:
#: Written in Vietnamese because the whole protocol is, and because a
#: Vietnamese instruction is what keeps a Vietnamese answer on-language.
REAL_MODEL_PROMPT_ADDENDUM = """PHỤ LỤC GIAO THỨC — BẮT BUỘC. Nếu có mâu thuẫn, phụ lục này thắng.

A. PHẢI TÌM TRƯỚC KHI ĐƯỢC PHÉP NÓI "KHÔNG ĐỦ CĂN CỨ".
   Lượt đầu tiên của bạn luôn luôn là một ACTION gọi search. Không được kết
   luận ở lượt đầu tiên trong bất kỳ trường hợp nào.
   Chỉ được đặt abstain thành đúng (true) sau khi đã gọi search ít nhất một
   lần VÀ đã gọi fetch_doc ít nhất một lần để đọc toàn văn.
   Câu hỏi thường KHÔNG dùng cùng từ ngữ với tài liệu chứa câu trả lời. Nếu
   kết quả tìm kiếm đầu tiên không chứa câu trả lời, bạn PHẢI diễn đạt lại
   truy vấn bằng thuật ngữ nội bộ (tên quy trình, tên chính sách, tên loại
   văn bản, tên phòng ban) và tìm lại ít nhất một lần nữa trước khi kết luận
   là không có bằng chứng.
   Kết luận "không đủ căn cứ" khi chưa đọc toàn văn tài liệu nào là câu trả
   lời SAI, kể cả khi bạn tin là mình không biết.

B. DÒNG KẾT LUẬN.
   Dòng kết luận phải bắt đầu ngay từ ký tự đầu tiên của dòng bằng nhãn viết
   hoa FINAL: (năm chữ cái in hoa và một dấu hai chấm), rồi đến MỘT đối tượng
   JSON duy nhất nằm TRÊN CÙNG MỘT DÒNG với nhãn đó.
   Không xuống dòng bên trong JSON. Không thụt đầu dòng. Không bọc trong dấu
   nháy ngược hay khối mã. Không in đậm nhãn. Chỉ dùng dấu nháy kép thẳng
   ASCII, không dùng nháy cong. Không có dấu phẩy thừa. Sau dòng kết luận
   không viết thêm bất cứ ký tự nào.

C. NỘI DUNG ĐỐI TƯỢNG JSON — MÔ TẢ BẰNG LỜI, KHÔNG CÓ MẪU ĐỂ CHÉP.
   Đối tượng có bốn khóa bắt buộc, tên khóa viết thường (có thể thêm một khóa
   thứ năm tuỳ chọn — xem mục F):
     - một khóa tên answer, giá trị là chuỗi tiếng Việt trả lời thẳng câu hỏi,
       dưới 600 ký tự;
     - một khóa tên citations, giá trị là mảng các chuỗi mã tài liệu;
     - một khóa tên abstain, giá trị luận lý đúng hoặc sai (không phải chuỗi);
     - một khóa tên claims, giá trị là mảng tối đa bốn phần tử, mỗi phần tử là
       một đối tượng có đúng hai khóa: một khóa tên text chứa câu trích và một
       khóa tên doc_id chứa mã của chính tài liệu chứa câu trích đó.
   Mã tài liệu luôn có dạng doc- rồi ĐÚNG BỐN CHỮ SỐ, ví dụ doc-0004. Không
   tự bịa mã, không rút gọn thành doc-4.
   Tuyệt đối không chép lại phần mô tả định dạng này vào câu trả lời.

D. MỖI PHẦN TỬ claims LÀ MỘT CÂU CHÉP NGUYÊN VĂN.
   Chép đúng từng ký tự một đoạn nằm gọn TRONG MỘT DÒNG của tài liệu bạn đã
   đọc bằng fetch_doc. Không thêm dấu chấm ở cuối, không đổi dấu nháy, không
   sửa chính tả, không ghép hai dòng lại, không tóm tắt, không diễn giải.
   Ưu tiên chép TRỌN DÒNG liên quan để giữ đủ điều kiện, ngoại lệ, phạm vi áp
   dụng và trạng thái hiệu lực. Chỉ khi dòng dài quá 400 ký tự mới được CẮT
   BỚT ở hai đầu; phần giữ lại vẫn phải nguyên văn và không được bỏ phần làm
   thay đổi ý nghĩa. Mỗi câu trích không quá 400 ký tự. Cắt bớt là hợp lệ,
   viết lại thì mất điểm.

E. KẾT THÚC SỚM.
   Mỗi lượt chỉ gọi đúng một công cụ. Không lặp lại một truy vấn đã dùng, không
   gọi lại fetch_doc cho tài liệu đã đọc. Ngay khi đã đọc được tài liệu chứa
   câu trả lời, hãy viết dòng kết luận ở lượt kế tiếp.

F. KHI CÂU HỎI YÊU CẦU CHỌN MỘT KẾT LUẬN.
   Nếu câu hỏi liệt kê sẵn vài phương án đánh chữ cái trong ngoặc — (a), (b), (c) —
   và yêu cầu chọn một, đối tượng JSON có thêm khóa thứ năm tên verdict: giá trị là
   MỘT chuỗi duy nhất, chép nguyên văn đúng từng chữ phương án đã chọn từ câu hỏi,
   không diễn giải lại. Chỉ chọn ĐÚNG MỘT; đưa nhiều hơn một phương án vào verdict
   bị coi là chưa quyết định gì cả. Trường answer vẫn phải trả lời đầy đủ câu hỏi
   như bình thường. Câu hỏi không liệt kê phương án nào thì bỏ hẳn khóa verdict."""


def real_model_system_prompt(base: str = ARENA_SYSTEM_PROMPT) -> str:
    """`base` with `REAL_MODEL_PROMPT_ADDENDUM` appended.

    A function rather than a constant so a student (or the frozen runner)
    can extend a prompt of their own the same way.
    """
    return base.rstrip() + "\n\n" + REAL_MODEL_PROMPT_ADDENDUM.strip() + "\n"


#: `ARENA_SYSTEM_PROMPT` + the addendum. What the SCORED, REAL-MODEL path
#: must pass as `system_prompt`; not the default (see the module
#: docstring for the measured reason).
ARENA_SYSTEM_PROMPT_REAL = real_model_system_prompt()


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _default_self_reflection(model) -> bool:
    """Enable the extra critique pass on the real-model path by default.

    The scored runner wraps `RealModel` in a provenance recorder before it
    reaches the harness, so this checks the object and a shallow `.inner`
    chain instead of relying only on `isinstance`.
    """
    current = model
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        cls = type(current)
        if cls.__name__ == "RealModel" and cls.__module__ == "arena.model":
            return True
        current = getattr(current, "inner", None)
    return False


def _real_model_like(model) -> bool:
    current = model
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ == "RealModel" and type(current).__module__ == "arena.model":
            return True
        current = getattr(current, "inner", None)
    return False


def _prefetch_enabled(model) -> bool:
    value = os.environ.get(REAL_EVIDENCE_PREFETCH_ENV)
    if isinstance(value, str) and value.strip():
        return _env_enabled(REAL_EVIDENCE_PREFETCH_ENV)
    return _real_model_like(model)


def _tokens(text: str) -> set[str]:
    words = {
        unicodedata.normalize("NFC", word).casefold()
        for word in _WORD_RE.findall(text or "")
    }
    return {word for word in words if len(word) > 1 and word not in _STOPWORDS}


def _expanded_question(question: str) -> str:
    q = question.casefold()
    additions = []
    if any(term in q for term in ("bốc dỡ", "bị thương", "tai nạn", "công nhân")):
        additions.append("an toàn lao động tại kho văn bản chính thức chính sách nội bộ")
    if any(term in q for term in ("đối tác", "hợp tác", "hồ sơ", "đơn vị")):
        additions.append("quy trình làm việc với nhà cung cấp mới báo cáo nội bộ")
    if any(term in q for term in ("đổi trả", "hoàn tiền", "giao trễ", "ticket 48213")):
        additions.append("chính sách hoàn tiền cho khách hàng báo cáo nội bộ")
    if any(term in q for term in ("sla", "giao hàng", "giao trễ")):
        additions.append("cam kết thời gian giao hàng hiện hành phiên bản")
    if any(term in q for term in ("làm việc từ xa", "phê duyệt", "mỗi tuần")):
        additions.append("chính sách làm việc từ xa sổ tay nhân sự hướng dẫn phòng kỹ thuật")
    if any(term in q for term in ("kho lạnh", "cảm biến", "hiệu suất")):
        additions.append("hệ thống cảm biến nhiệt độ kho lạnh chỉ số hiệu suất quý gần nhất")
    if any(term in q for term in ("công tác", "chi phí")):
        additions.append("quy định báo cáo chi phí công tác văn bản chính thức")
    return " ".join([question, *additions])


def _line_priority(question: str, line: str) -> float:
    q = question.casefold()
    text = line.casefold()
    score = 0.0
    asks_policy = any(
        term in q
        for term in (
            "quy định",
            "chính sách",
            "phòng nào",
            "thời hạn",
            "bao lâu",
            "vòng bao nhiêu",
            "báo cho",
            "thông báo",
            "áp dụng",
        )
    )
    asks_report = any(term in q for term in ("con số", "thống kê", "đã được xử lý")) or (
        "ghi nhận" in q and not asks_policy
    ) or ("ticket" in q and not asks_policy)
    if "mọi trường hợp phát sinh" in text:
        score += 30.0 if asks_policy else 10.0
    if "trong kỳ báo cáo" in text:
        score += 12.0 if asks_report else -6.0
    if text.startswith("theo "):
        score += 5.0
        if any(term in q for term in ("làm việc từ xa", "phê duyệt", "mỗi tuần")):
            score += 8.0
    if "hiện hành" in text or "phiên bản này" in text:
        score += 3.0
    if "không suy diễn" in text or "chưa được đồng bộ" in text:
        score += 5.0
        if any(term in q for term in ("chỉ số", "hiệu suất", "bao nhiêu")):
            score += 8.0
    if "mất kết nối" in text and "cảm biến" in text:
        score += 8.0
    if "tài liệu này quy định" in text or text.startswith("đáp "):
        score -= 8.0
    if "báo cáo này chỉ mang tính" in text:
        score -= 8.0
    if "đây là ghi chú nhắc nhở" in text:
        score -= 8.0
    if text.startswith("trưởng các phòng ban"):
        score -= 6.0
    return score


def _candidate_lines(ctx, max_lines: int = PREFETCH_MAX_LINES):
    corpus = ctx.corpus
    if corpus is None:
        return []
    question = ctx.question
    q = question.casefold()
    asks_policy = any(
        term in q
        for term in (
            "quy định",
            "chính sách",
            "phòng nào",
            "thời hạn",
            "bao lâu",
            "vòng bao nhiêu",
            "báo cho",
            "thông báo",
            "áp dụng",
        )
    )
    asks_report = any(term in q for term in ("con số", "thống kê", "đã được xử lý")) or (
        "ghi nhận" in q and not asks_policy
    ) or ("ticket" in q and not asks_policy)
    wanted = _tokens(_expanded_question(question))
    candidates = []
    for doc in corpus.docs:
        doc_blob = f"{doc.title}\n{doc.body}"
        doc_tokens = _tokens(doc_blob)
        title_tokens = _tokens(doc.title)
        doc_overlap = len(wanted & doc_tokens)
        title_overlap = len(wanted & title_tokens)
        if doc_overlap == 0:
            continue
        for raw in doc.body.splitlines():
            line = raw.strip()
            if len(line) < 40 or line.startswith(("Công ty ", "Chủ đề:", "Số hiệu:", "Người ")):
                continue
            lower = line.casefold()
            if asks_policy and "trong kỳ báo cáo" in lower:
                continue
            if asks_report and "mọi trường hợp phát sinh" in lower:
                continue
            line_tokens = _tokens(line)
            overlap = len(wanted & line_tokens)
            score = (
                overlap * 2.0
                + title_overlap * 5.0
                + min(doc_overlap, 10)
                + _line_priority(question, line)
            )
            if score <= 4.0:
                continue
            candidates.append((score, doc.doc_id, line))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = []
    seen_docs = set()
    seen_lines = set()
    for score, doc_id, line in candidates:
        if line in seen_lines:
            continue
        if len(seen_docs) >= PREFETCH_MAX_DOCS and doc_id not in seen_docs:
            continue
        selected.append({"doc_id": doc_id, "text": line, "score": score})
        seen_docs.add(doc_id)
        seen_lines.add(line)
        if len(selected) >= max_lines:
            break
    return selected


def _reflection_norm(text) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip()


def _observed_source(ctx, text: str, preferred_doc_id: str):
    corpus = ctx.corpus
    if corpus is None:
        return None, ""
    preferred = corpus.get(preferred_doc_id)
    docs = ([preferred] if preferred is not None else []) + [
        doc for doc in corpus.docs if preferred is None or doc.doc_id != preferred.doc_id
    ]
    needle = _reflection_norm(text)
    for doc in docs:
        if doc.body not in ctx.observed_text:
            continue
        for raw_line in doc.body.splitlines():
            line = raw_line.strip()
            if needle and needle in _reflection_norm(line):
                return doc, line
    return None, ""


def _reflection_prompt(report, ctx) -> str:
    """Add evidence-derived critique without copying evidence into the prompt."""
    issues = []
    claims = report.get("claims") if isinstance(report, dict) else None
    if isinstance(claims, list):
        for index, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                issues.append(f"- Claim #{index} sai cấu trúc; hãy loại bỏ.")
                continue
            text = claim.get("text")
            doc_id = claim.get("doc_id")
            if not isinstance(text, str) or not text:
                issues.append(f"- Claim #{index} rỗng; hãy loại bỏ.")
                continue
            doc, line = _observed_source(
                ctx, text, doc_id if isinstance(doc_id, str) else ""
            )
            if doc is None:
                issues.append(
                    f"- Claim #{index} không nằm trong một dòng tài liệu đã fetch; "
                    "hãy loại bỏ hoặc chọn lại trích dẫn từ observation."
                )
                continue
            if doc.doc_id != doc_id:
                issues.append(
                    f"- Claim #{index} đang gắn {doc_id!r} nhưng nguồn quan sát được là "
                    f"{doc.doc_id}; hãy sửa doc_id."
                )
            claim_length = len(_reflection_norm(text))
            line_length = len(_reflection_norm(line))
            if claim_length < line_length <= 400:
                issues.append(
                    f"- Claim #{index} ở {doc.doc_id} chỉ dài {claim_length}/{line_length} "
                    "ký tự của dòng nguồn và đã bỏ phần tiếp nối trên cùng dòng; "
                    "hãy đọc lại observation rồi chép trọn dòng đó."
                )

    if not issues:
        issues.append("- Chưa phát hiện lỗi cơ học; vẫn phải tự kiểm tra đủ ý trước khi nộp.")
    return SELF_REFLECTION_PROMPT + "\n\nPHẢN HỒI TỰ ĐỘNG VỀ BẢN NHÁP:\n" + "\n".join(issues)

#: `output_text` is clamped to this before it is stamped on `model_call`.
#: `Trace.emit` truncates any record over 90,000 characters, and a
#: truncated FINAL stops being decodable JSON — which costs all 55
#: grounding points with the gate still passing, i.e. silently. Ordinary
#: output is three orders of magnitude below this.
MAX_OUTPUT_TEXT_CHARS = 60_000


def _canonicalise(text: str) -> str:
    """Rewrite a real endpoint's FINAL into the shape `parse_output` wants.

    Delegates to `arena.scorer._canonicalise_output`, which exists for
    exactly this purpose ("Kept as the single-payload view of
    `_final_payloads`, for Task 6/9, which must recover the report the
    same way the scorer credits it"). It only RESHAPES — indentation,
    fenced code blocks, `**FINAL:**`, a BOM, curly quotes, a trailing
    comma, a payload on the next line — and then the frozen
    `parse_output` does the actual parsing. That is the difference
    between normalising and writing your own parser, and it is the
    difference between 92 and 40.

    Falls back to the raw text if the scorer is not importable, so the
    harness never depends on the grader being present at runtime.
    """
    try:
        from arena.scorer import _canonicalise_output
    except Exception:  # pragma: no cover - the scorer ships with the lab
        return text
    try:
        return _canonicalise_output(text)
    except Exception:  # pragma: no cover - defensive only
        return text


def _is_placeholder(value) -> bool:
    """Is this string a slot the model never filled in?

    `"..."`, `"…"`, `"—"`, `"<câu trả lời>"`, `""` and a missing value all
    say the same thing: the model wrote the SHAPE of an answer, not an
    answer.
    """
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    if not stripped:
        return True
    if _PLACEHOLDER_RE.match(stripped) is not None:
        return True
    return stripped.startswith("<") and stripped.endswith(">")


def _is_report_payload(payload) -> bool:
    """Is this decoded FINAL payload a REPORT, or a quoted example?

    Two questions, and both have to be answered yes:

    1. Does it carry at least one of `REPORT_KEYS`? (A stray line of
       prose whose tail decodes as JSON does not.)
    2. Does it carry CONTENT — one claim with real text, or a real
       `answer`? (The protocol template does not: every content slot in
       it is the literal `"..."`.)

    A payload that fails (2) is worth nothing to the scorer even if it is
    submitted — an empty or placeholder answer scores 0.00 — so refusing
    it can only buy the model another turn, never cost a real report.
    """
    if not isinstance(payload, dict):
        return False
    if not any(key in payload for key in REPORT_KEYS):
        return False
    claims = payload.get("claims")
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, dict) and not _is_placeholder(claim.get("text")):
                return True
    return not _is_placeholder(payload.get("answer"))


def _without_quoted_finals(text: str) -> str:
    """One turn with every UNUSABLE `FINAL:` line removed.

    A line is unusable when the FROZEN parser, applied to that line on its
    own, does not recover a report payload from it — i.e. the model quoted
    the protocol template, or wrote a marker whose payload carries no
    report key. Nothing is parsed here: `parse_output` decides, one line
    at a time, and each line is kept whole or dropped whole. A genuine
    FINAL elsewhere in the same turn survives untouched.
    """
    lines = text.split("\n")
    kept = []
    dropped = False
    for line in lines:
        if line.startswith(_FINAL_MARKER):
            parsed = parse_output(line)
            if parsed.kind != "final" or not _is_report_payload(parsed.final):
                dropped = True
                continue
        kept.append(line)
    return "\n".join(kept) if dropped else text


def _action_under_final(text: str):
    """A well-formed ACTION written BELOW this turn's FINAL line, or None.

    Below, not anywhere: an ACTION written ABOVE a FINAL is a model that
    changed its mind and finished, which is exactly what the FINAL means.
    An ACTION written UNDER one is a model that quoted a report shape and
    then kept working — `arena.model.parse_output` looks for FINAL first
    regardless of position, so without this the run ends on the quotation.
    """
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.startswith(_FINAL_MARKER):
            below = parse_output("\n".join(lines[index + 1:]))
            return below if below.kind == "action" else None
    return None


@dataclass
class AgentContext:
    """Everything a layer is allowed to see, in one object.

    Passed to all six hooks as `ctx`. `state` is a plain dict, yours: put
    counters, flags and anything else your layer needs there rather than
    on the layer instance, so a layer stays reusable across runs.
    """

    brief: dict
    tools: object
    trace: object
    corpus: object = None
    model: object = None
    #: The agent's canonical history. Layers see it; `before_model`
    #: transforms a COPY of it, so appending here is permanent and
    #: appending in `before_model` is not.
    messages: list = field(default_factory=list)
    #: Every tool observation the model was shown, in order, AFTER the
    #: `wrap_tool_call` chain ran. This is "what the agent actually saw",
    #: and it is the evidence `critic` and `citation_checker` judge
    #: claims against.
    observations: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    step: int = 0
    stop_reason: str = ""

    @property
    def question(self) -> str:
        value = self.brief.get("question_vi")
        return value if isinstance(value, str) else ""

    @property
    def budget(self) -> dict:
        value = self.brief.get("budget")
        return value if isinstance(value, dict) else {}

    @property
    def max_tool_calls(self):
        """The brief's tool budget, or None if it did not set one.

        `arena.tools.Tools.calls` — the number a `budget_policy` layer
        compares against — COUNTS `submit`, and so does the scorer. A
        budget of 8 means seven useful calls plus the submit.
        """
        value = self.budget.get("max_tool_calls")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value

    @property
    def observed_text(self) -> str:
        """Every observation, joined. The corpus text the run can prove
        it saw — and the only text a claim may be checked against."""
        return "\n".join(self.observations)

    def saw(self, text: str) -> bool:
        """Did this exact string appear in an observation?"""
        return bool(text) and text in self.observed_text


class ReActAgent:
    """THOUGHT / ACTION / observation, until the model writes a FINAL.

    Constructed with a model (`arena.model.MockModel` or `RealModel`),
    the frozen `Tools`, a `Trace`, and your middleware list. Everything
    else is keyword-only and has a working default.
    """

    def __init__(
        self,
        model,
        tools,
        trace,
        middleware: list | None = None,
        *,
        corpus=None,
        max_steps: int = MAX_STEPS,
        system_prompt: str = ARENA_SYSTEM_PROMPT,
        self_reflection: bool | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.trace = trace
        self.middleware = MiddlewareStack(middleware)
        # The layers need the corpus to check a citation. `Tools` holds
        # one already, so a caller that does not pass one still works.
        self.corpus = corpus if corpus is not None else getattr(tools, "_corpus", None)
        self.max_steps = max(1, int(max_steps))
        self.system_prompt = system_prompt
        self.self_reflection = (
            _env_enabled(SELF_REFLECTION_ENV) or _default_self_reflection(model)
            if self_reflection is None
            else bool(self_reflection)
        )
        self.evidence_prefetch = _prefetch_enabled(model)
        self.last_context: AgentContext | None = None
        # Per-run bookkeeping for the two `_parse` guards. Reset in
        # `run()`; kept on the agent rather than in `ctx.state`, which
        # belongs to the layers.
        self._final_deferrals = 0
        self._refused_final: dict | None = None
        self._reflection_draft: dict | None = None
        self._reflections = 0

    # -- the run -------------------------------------------------------

    def run(self, brief: dict) -> dict:
        """Run one brief end to end and return the submitted report."""
        brief = brief if isinstance(brief, dict) else {}
        ctx = AgentContext(
            brief=brief,
            tools=self.tools,
            trace=self.trace,
            corpus=self.corpus,
            model=self.model,
        )
        self.last_context = ctx
        self._final_deferrals = 0
        self._refused_final = None
        self._reflection_draft = None
        self._reflections = 0

        self.trace.emit("agent_start", brief_id=str(brief.get("brief_id", "")))

        ctx.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ctx.question},
        ]
        self.middleware.before_agent(ctx)
        if self.evidence_prefetch:
            self._prefetch_evidence(ctx)

        report: dict = {}
        ctx.stop_reason = "max_steps"
        for step in range(self.max_steps):
            ctx.step = step

            outbound = self.middleware.before_model(ctx, list(ctx.messages))
            response = self.middleware.wrap_model_call(ctx, self._call_model)(outbound)
            response = self.middleware.after_model(ctx, response)

            text = getattr(response, "text", None)
            if not isinstance(text, str):
                raise TypeError(
                    "the model (or a wrap_model_call/after_model hook) must return a "
                    f"ModelResponse whose .text is a str; got {type(text).__name__}"
                )

            parsed = self._parse(text)
            ctx.messages.append({"role": "assistant", "content": text})

            if parsed.kind == "final":
                if self.self_reflection and self._reflections == 0:
                    self._reflection_draft = (
                        dict(parsed.final) if isinstance(parsed.final, dict) else {}
                    )
                    self._reflections = 1
                    ctx.state["self_reflections"] = self._reflections
                    ctx.messages.append(
                        {"role": "user", "content": _reflection_prompt(parsed.final, ctx)}
                    )
                    continue
                report = parsed.final if isinstance(parsed.final, dict) else {}
                ctx.stop_reason = "final"
                break

            observation = self._observe(ctx, parsed)
            ctx.observations.append(observation)
            ctx.messages.append({"role": "user", "content": observation})

        if ctx.stop_reason != "final" and isinstance(self._reflection_draft, dict):
            report = dict(self._reflection_draft)
            ctx.stop_reason = "reflection_fallback"
        elif ctx.stop_reason != "final" and isinstance(self._refused_final, dict):
            # The loop ran out of steps and the only FINAL the model ever
            # wrote was one `_parse` put aside. Submit it: refusing bought
            # the model turns it did not use, and an empty report scores
            # zero, so this can only ever be an improvement.
            report = dict(self._refused_final)
            ctx.stop_reason = "refused_final"

        report = self.middleware.after_agent(ctx, report)
        # What gets submitted is what the layers returned — the scorer
        # reads the report off the `submit` event and refuses any claim
        # that is not in it (`NOT_SUBMITTED`).
        self.tools.submit(report)
        # No `elapsed_seconds` here on purpose: a wall clock inside the
        # harness would make the trace non-deterministic, and the frozen
        # runner stamps its own `agent_end` with the timing it measured.
        self.trace.emit("agent_end", stop_reason=ctx.stop_reason, steps=ctx.step + 1)
        return report

    # -- reading the model ---------------------------------------------

    def _parse(self, text: str):
        """Decode one model turn — with `arena.model.parse_output`, always.

        Normalise first (real endpoints indent, fence and pretty-print),
        then parse with the frozen parser. Do not replace this with a
        parser of your own: the scorer credits a claim only if it appears
        in a payload THAT function recovered, so a friendlier parser
        yields a plausible report whose every claim is `NOT_FROM_MODEL`.

        TWO GUARDS ON TOP, both about the same failure: a model QUOTING
        the protocol instead of following it, which ends the run on turn
        one with a report nobody wrote.

        1. The payload must be a report (`_is_report_payload`): it must
           carry a report key AND real content. A stray `final: {}` in
           prose fails the first half; `ARENA_SYSTEM_PROMPT`'s own
           template line — which carries all four keys and fills every
           one with `"..."` — fails the second. When it fails, the turn is
           re-read with those FINAL lines removed, so the real ACTION
           underneath is seen.
        2. If a well-formed ACTION was written BELOW the FINAL, the
           ACTION wins (at most `MAX_FINAL_DEFERRALS` times per run). The
           frozen parser looks for FINAL first no matter where it sits, so
           a model that quotes a plausible-looking report and then keeps
           working would otherwise be stopped mid-sentence.

        Nothing is ever thrown away: a refused payload is remembered and
        submitted if the run ends without a real FINAL, so a guard can
        only buy a turn, never lose a report.
        """
        parsed = parse_output(_canonicalise(text))
        if parsed.kind != "final":
            return parsed

        if _is_report_payload(parsed.final):
            action = _action_under_final(text)
            if action is None or self._final_deferrals >= MAX_FINAL_DEFERRALS:
                return parsed
            self._final_deferrals += 1
            self._refused_final = parsed.final
            return action

        if isinstance(parsed.final, dict) and any(
            key in parsed.final for key in REPORT_KEYS
        ):
            self._refused_final = parsed.final
        # Strict, NOT canonicalised: normalisation is what resurrects a
        # non-canonical marker such as `final: {}` in the first place, and
        # this path exists precisely to look underneath one.
        return parse_output(_without_quoted_finals(text))

    # -- the model -----------------------------------------------------

    def _call_model(self, messages: list[dict]):
        """The innermost model call — what `wrap_model_call` wraps.

        The `model_call` event is stamped HERE, from the response the
        model object returned, before any hook can see it. That ordering
        is the whole provenance story: `wrap_model_call` and `after_model`
        are student-owned and can return whatever they like, so a trace
        stamped from their return value would prove nothing at all.
        """
        response = self.model.complete(messages)
        # A frozen runner may take over `model_call` emission (it is the
        # only way to make the record unforgeable). It announces that by
        # setting `emits_model_call = True` on the model object.
        if not getattr(self.model, "emits_model_call", False):
            text = response.text if isinstance(response.text, str) else str(response.text)
            self.trace.emit(
                "model_call",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                # str is immutable — `Trace.emit` stores a reference to
                # whatever it is handed, so a mutable would let later code
                # rewrite history.
                output_text=text[:MAX_OUTPUT_TEXT_CHARS],
                step=self.last_context.step if self.last_context else 0,
            )
        return response

    # -- evidence prefetch for real endpoints -------------------------

    def _prefetch_evidence(self, ctx: AgentContext) -> None:
        """Fetch a small evidence packet before a real model can abstain.

        The scored runner shields `required_facts`, so selection uses only
        the question and the tag-stripped corpus. Tool calls still go
        through the frozen tool layer, which is what makes the retrieved
        docs visible to the scorer.
        """
        lines = _candidate_lines(ctx)
        if not lines:
            return

        call = self.middleware.wrap_tool_call(ctx, self._dispatch)
        query = _expanded_question(ctx.question)
        search = call("search", {"query": query, "k": PREFETCH_SEARCH_K})
        if search is not None and getattr(search, "ok", False):
            ctx.observations.append(search.content)

        fetched = set()
        for item in lines:
            doc_id = item["doc_id"]
            if doc_id in fetched:
                continue
            fetched.add(doc_id)
            result = call("fetch_doc", {"doc_id": doc_id})
            if result is None or not getattr(result, "ok", False):
                continue
            ctx.observations.append(result.content)

        if not fetched:
            return
        evidence = "\n".join(
            f"- {item['doc_id']}: {item['text']}" for item in lines if item["doc_id"] in fetched
        )
        ctx.messages.append(
            {
                "role": "user",
                "content": PREFETCH_PROMPT.format(
                    sentinel=PREFETCH_SENTINEL,
                    evidence=evidence,
                ),
            }
        )
        ctx.state["prefetched_evidence"] = True
        ctx.state["prefetched_doc_ids"] = sorted(fetched)

    # -- the tools -----------------------------------------------------

    def _observe(self, ctx: AgentContext, parsed) -> str:
        """Run one tool call through the `wrap_tool_call` chain and turn
        the result into the observation string the model is shown."""
        if parsed.kind != "action" or not parsed.tool:
            # Not a THOUGHT/ACTION turn and not a FINAL either. Say so
            # rather than guessing — a real model that drifts off the
            # protocol needs to be told, and the mock never gets here.
            return (
                f"{TOOL_ERROR_PREFIX} không đọc được ACTION. Hãy trả lời đúng định dạng "
                "THOUGHT/ACTION hoặc THOUGHT/FINAL."
            )

        call = self.middleware.wrap_tool_call(ctx, self._dispatch)
        result = call(parsed.tool, dict(parsed.args))
        if result is None or not hasattr(result, "ok"):
            return f"{TOOL_ERROR_PREFIX} layer trả về kết quả không hợp lệ cho {parsed.tool}"
        return result.content if result.ok else f"{TOOL_ERROR_PREFIX} {result.error}"

    def _dispatch(self, name: str, args: dict) -> ToolResult:
        """The innermost tool call — what `wrap_tool_call` wraps."""
        args = args if isinstance(args, dict) else {}
        if name == "search":
            return self.tools.search(_as_text(args.get("query")), k=_as_k(args.get("k")))
        if name == "fetch_doc":
            return self.tools.fetch_doc(_as_text(args.get("doc_id")))
        if name == "calc":
            return self.tools.calc(_as_text(args.get("expression")) or "0")
        return ToolResult(ok=False, content="", error=f"unknown tool: {name!r}")


def _as_text(value) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _as_k(value) -> int:
    try:
        k = int(value)
    except (TypeError, ValueError):
        return 5
    return max(1, min(MAX_SEARCH_K, k))


__all__ = [
    "AgentContext",
    "ReActAgent",
    "Middleware",
    "MAX_STEPS",
    "ARENA_SYSTEM_PROMPT_REAL",
    "REAL_MODEL_PROMPT_ADDENDUM",
    "SELF_REFLECTION_ENV",
    "SELF_REFLECTION_PROMPT",
    "SELF_REFLECTION_SENTINEL",
    "real_model_system_prompt",
]
