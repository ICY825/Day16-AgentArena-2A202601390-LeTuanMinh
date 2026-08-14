#!/usr/bin/env python3
"""Chạy vòng LUYỆN TẬP: agent của bạn + bộ brief công khai, rồi chấm điểm.

    python3 scripts/run_practice.py                  # mock, 8 brief công khai
    python3 scripts/run_practice.py --brief pub-01-sla-hien-hanh
    python3 scripts/run_practice.py --layers none    # baseline, không lớp nào
    python3 scripts/run_practice.py --model real     # cần ARENA_* trong env

Kết quả in ra màn hình VÀ ghi vào một file điểm JSON (mặc định
`runs/practice.json`) — file đó là thứ `scripts/leaderboard.py` đọc.

BẢNG ĐIỂM LUYỆN TẬP CHỈ MANG TÍNH THAM KHẢO
===========================================

Bộ brief công khai được viết để bạn gỡ lỗi, không phải để xếp hạng: đáp án
của nó nằm trong chính câu hỏi + corpus, nên một harness 30 dòng "trích
dòng dài nhất của top-5 tài liệu" đạt điểm rất cao ở đây và gần như bằng 0
ở vòng có tính điểm. Vòng có tính điểm chạy trên BỘ BRIEF RIÊNG, với model
thật, do giảng viên thực thi. Hãy dùng vòng luyện tập để kiểm tra rằng năm
lớp của bạn thực sự hoạt động — đừng tối ưu con số ở đây.

TẤT CẢ ĐỀU CHẠY QUA `arena/runner.py` (runner đóng băng). Điểm bạn thấy ở
đây được tính đúng bằng cơ chế của vòng thi thật: cùng một hàm chấm, cùng
một quy tắc provenance, cùng một cách ghi trace.

FILE ĐIỂM GHI CẢ CHẨN ĐOÁN, KHÔNG CHỈ CON SỐ
============================================

Bộ chấm ĐÃ TÍNH sẵn lý do cho từng điểm số — verdict của từng claim,
recall/precision, dữ kiện nào thiếu, cổng trace hỏng ở đâu — và trả về
trong `Score.detail`. Trước đây chỗ đó bị vứt đi ngay sau khi in ra một
dòng `G/S/E`, nên bạn thấy "G 6.9 / 55" mà không có cách nào biết mình
mất điểm vì THIẾU dữ kiện, vì PARAPHRASE, vì TRÍCH SAI TÀI LIỆU hay vì
một claim mô hình chưa từng viết.

Bây giờ mỗi dòng trong `runs/*.json` mang thêm hai khoá:

    "detail"     — nguyên văn `Score.detail` của bộ chấm (không sửa gì).
    "diagnostic" — những thứ `detail` cố tình bỏ đi mà việc chẩn đoán
                   cần: chữ của từng claim bạn nộp, danh sách tài liệu
                   lượt chạy thật sự đọc, phần FINAL mô hình đã viết,
                   số lần tool lỗi.

Mọi khoá cũ giữ nguyên — `scripts/leaderboard.py` và pytest đọc đúng như
trước. Để ĐỌC chỗ dữ liệu đó bằng tiếng người:

    python3 scripts/selfeval.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from arena.briefs import (  # noqa: E402
    PUBLIC_BRIEFS_PATH,
    load_briefs,
    load_public_briefs,
)
from arena.corpus import Corpus  # noqa: E402
from arena.model import MockModel, RealModel, RealModelError  # noqa: E402
from arena.runner import (  # noqa: E402
    RUNNER_VERSION,
    PreflightError,
    RunnerConfig,
    assert_provenance,
    derive_seed,
    run_brief,
    score_result,
)

SCHEMA = "arena-scores/1"
DEFAULT_OUT = LAB_ROOT / "runs" / "practice.json"
DEFAULT_ENV = LAB_ROOT / ".env"

#: Thứ tự cài đặt năm lớp — xem `harness/middleware.py`.
STACK_ORDER = (
    "injection_guard",
    "critic",
    "citation_checker",
    "budget_policy",
    "retry",
)


def load_dotenv(path: Path = DEFAULT_ENV) -> None:
    """Load simple KEY=VALUE lines without overwriting real environment vars."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _student_layers(names):
    from harness.layers.budget_policy import BudgetPolicy
    from harness.layers.citation_checker import CitationChecker
    from harness.layers.critic import Critic
    from harness.layers.injection_guard import InjectionGuard
    from harness.layers.retry import Retry

    classes = {
        "injection_guard": InjectionGuard,
        "critic": Critic,
        "citation_checker": CitationChecker,
        "budget_policy": BudgetPolicy,
        "retry": Retry,
    }
    return [classes[name]() for name in STACK_ORDER if name in names]


def build_middleware(spec: str):
    """`spec` là "all", "none", hoặc danh sách tên cách nhau bởi dấu phẩy."""
    spec = (spec or "all").strip().lower()
    if spec in ("none", "", "-"):
        return [], []
    if spec == "all":
        names = set(STACK_ORDER)
    else:
        names = {part.strip() for part in spec.split(",") if part.strip()}
        unknown = names - set(STACK_ORDER)
        if unknown:
            raise SystemExit(
                f"Không biết lớp: {sorted(unknown)}. Hợp lệ: {list(STACK_ORDER)}"
            )
    return _student_layers(names), [name for name in STACK_ORDER if name in names]


def build_model(kind: str, corpus: Corpus, seed: int, timeout: float):
    if kind == "mock":
        return MockModel(corpus=corpus, seed=seed)
    try:
        return RealModel.from_env(timeout=timeout)
    except RealModelError as exc:
        raise SystemExit(f"\n{exc}\n")


def load_brief_set(name: str):
    if name == "public":
        return load_public_briefs(), "public"
    if name == "private":
        raise SystemExit(
            "Bộ brief có tính điểm KHÔNG nằm trong bản phát cho sinh viên: nó chỉ tồn "
            "tại trên máy giảng viên và chỉ chạy ở vòng chấm điểm. Hãy luyện tập với "
            "`--briefs public`, hoặc trỏ `--briefs` tới một file brief bạn tự viết."
        )
    path = Path(name)
    if not path.exists():
        raise SystemExit(f"Không tìm thấy bộ brief: {name}")
    return load_briefs(path), path.stem


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(max(0.0, min(100.0, value)) / 100 * width))
    return "█" * filled + "·" * (width - filled)


# ---------------------------------------------------------------------------
# Chẩn đoán: giữ lại thứ bộ chấm đã tính, thay vì vứt đi
# ---------------------------------------------------------------------------

#: Trần độ dài cho từng trường văn bản ghi vào file điểm. Đủ rộng cho một
#: claim hợp lệ (`scorer.MAX_CLAIM_CHARS` = 500) và cho một `answer`
#: đúng chuẩn (`scorer.ANSWER_SCAN_CHARS` = 1500), nhưng không để một
#: lượt chạy hỏng bơm hàng megabyte vào `runs/*.json`.
DIAG_CLAIM_CHARS = 600
DIAG_ANSWER_CHARS = 1600
DIAG_MODEL_TEXT_CHARS = 8000
DIAG_MAX_CLAIMS = 24


def _plain(value):
    """Bản sao chắc chắn JSON-hoá được. `Score.detail` do bộ chấm dựng
    nên đã sạch, nhưng file điểm không được phép hỏng vì một giá trị lạ:
    một lần chạy luyện tập mất cả bảng điểm thì đắt hơn nhiều so với một
    trường bị ép thành chuỗi."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return {}


def _diagnostic(result, corpus) -> dict:
    """Những gì `Score.detail` CỐ TÌNH bỏ đi mà việc chẩn đoán vẫn cần.

    Bộ chấm cắt `text` khỏi từng verdict claim (nó chỉ cần verdict để
    tính điểm). Nhưng "claim #2 HALLUCINATED" mà không có chữ của claim
    #2 thì không sửa được gì cả — nhất là với ca hay gặp nhất trong lab
    này: claim CHỈ LỆCH MỘT DẤU CHẤM so với một dòng của tài liệu. Vì
    vậy chỗ này ghi lại:

    * `claims` — chữ và doc_id của từng claim ĐÃ NỘP, để đối chiếu
      nguyên văn với corpus (xem `scripts/selfeval.py`).
    * `retrieved_doc_ids` — tài liệu lượt chạy CHỨNG MINH được là đã
      đọc, đọc bằng chính hàm bộ chấm dùng, nên "UNRETRIEVED" giải
      thích được thay vì chỉ là một nhãn.
    * `model_final_text` — phần FINAL mô hình thật sự viết, đã chuẩn
      hoá. Đây là thứ quyết định `NOT_FROM_MODEL`: có nó thì phân biệt
      được "mô hình không viết gì" với "một layer của bạn đã viết lại
      chữ của claim".
    * `tool_failures` — lỗi tool ngẫu nhiên đã bắn mấy lần, tức bẫy
      `flaky` có thực sự xuất hiện trong lượt này không.

    Không có gì ở đây là điểm số, và không có gì ở đây quay ngược lại
    ảnh hưởng tới điểm: đây là bản ghi chép, không phải một chiều chấm.
    """
    diag: dict = {
        "claims": [],
        "answer": "",
        "abstain": False,
        "citations": [],
        "retrieved_doc_ids": [],
        "model_final_text": "",
        "model_final_chars": 0,
        "model_final_truncated": False,
        "tool_failures": 0,
        "tool_failures_by_name": {},
        "flaky_modes": {},
    }

    report = result.report if isinstance(result.report, dict) else {}
    try:
        raw_claims = report.get("claims")
        if isinstance(raw_claims, list):
            for index, claim in enumerate(raw_claims[:DIAG_MAX_CLAIMS]):
                if isinstance(claim, dict):
                    text = claim.get("text")
                    doc_id = claim.get("doc_id")
                else:
                    text, doc_id = None, None
                diag["claims"].append(
                    {
                        "index": index,
                        "doc_id": doc_id if isinstance(doc_id, str) else "",
                        "text": (text if isinstance(text, str) else "")[:DIAG_CLAIM_CHARS],
                        "malformed": not isinstance(claim, dict),
                    }
                )
        answer = report.get("answer")
        if isinstance(answer, str):
            diag["answer"] = answer[:DIAG_ANSWER_CHARS]
        diag["abstain"] = report.get("abstain") is True
        citations = report.get("citations")
        if isinstance(citations, list):
            diag["citations"] = [c for c in citations if isinstance(c, str)][:32]
    except Exception:  # một report thù địch không được làm hỏng file điểm
        pass

    # Đọc trace bằng CHÍNH hàm bộ chấm dùng, không viết lại: nếu hai bên
    # lệch nhau thì phần chẩn đoán sẽ nói dối đúng vào lúc nó được tin
    # nhất. `_read_trace` là nội bộ của module đóng băng, nên lời gọi
    # được bọc lại — thiếu chẩn đoán còn hơn mất cả lượt chạy.
    try:
        from arena.scorer import _read_trace

        facts = _read_trace(result.trace_jsonl, corpus)
        diag["retrieved_doc_ids"] = sorted(facts.retrieved)
        model_text = facts.model_text or ""
        diag["model_final_chars"] = len(model_text)
        diag["model_final_truncated"] = len(model_text) > DIAG_MODEL_TEXT_CHARS
        diag["model_final_text"] = model_text[:DIAG_MODEL_TEXT_CHARS]
    except Exception:
        pass

    try:
        for line in result.trace_jsonl.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if not isinstance(record, dict) or record.get("event") != "tool_call":
                continue
            mode = record.get("flaky_mode")
            if isinstance(mode, str) and mode and mode != "none":
                diag["flaky_modes"][mode] = diag["flaky_modes"].get(mode, 0) + 1
            if record.get("ok") is False:
                name = record.get("name")
                name = name if isinstance(name, str) else "?"
                diag["tool_failures"] += 1
                diag["tool_failures_by_name"][name] = (
                    diag["tool_failures_by_name"].get(name, 0) + 1
                )
    except Exception:
        pass

    return diag


def main(argv=None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Chạy vòng luyện tập Agent Arena và chấm điểm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--briefs", default="public",
                        help="public | đường dẫn tới một file brief bạn tự viết")
    parser.add_argument("--brief", default=None, help="chỉ chạy một brief_id")
    parser.add_argument("--model", default="mock", choices=("mock", "real"))
    parser.add_argument("--layers", default="all",
                        help='"all", "none", hoặc "critic,retry"')
    parser.add_argument("--seed", type=int, default=11, help="seed gốc (mỗi brief +1)")
    parser.add_argument("--corpus-seed", type=int, default=None,
                        help="mặc định lấy từ file brief")
    parser.add_argument("--no-flaky", action="store_true",
                        help="tắt lỗi ngẫu nhiên của tool (chỉ để gỡ lỗi)")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="trần thời gian mỗi lần chạy")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="ngân sách output mỗi lần gọi model")
    parser.add_argument("--prompt-addendum", action="store_true",
                        help="thêm ràng buộc 'phải search trước khi abstain' vào system prompt")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="file điểm JSON")
    parser.add_argument("--entry", default=None, help="tên đội / bài nộp")
    parser.add_argument("--tag", default="entry", choices=("entry", "baseline"),
                        help='"baseline" đánh dấu file này là mốc so sánh cho leaderboard')
    parser.add_argument("--strict", action="store_true",
                        help="thoát khác 0 nếu không có model_call nào chứa FINAL")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    briefs, set_name = load_brief_set(args.briefs)
    if args.brief:
        briefs = [b for b in briefs if b.get("brief_id") == args.brief]
        if not briefs:
            raise SystemExit(f"Không có brief nào tên {args.brief!r}")

    corpus_seed = args.corpus_seed
    if corpus_seed is None:
        source = PUBLIC_BRIEFS_PATH if set_name == "public" else Path(args.briefs)
        try:
            corpus_seed = int(json.loads(source.read_text(encoding="utf-8"))["corpus_seed"])
        except Exception:
            corpus_seed = 42
    corpus = Corpus.generate(seed=corpus_seed)

    _probe, layer_names = build_middleware(args.layers)
    del _probe  # a FRESH stack per brief; no state may leak between runs
    config = RunnerConfig(
        flaky=not args.no_flaky,
        prompt_addendum=args.prompt_addendum,
        **({"wall_clock_seconds": args.max_seconds} if args.max_seconds else {}),
        **({"max_tokens": args.max_tokens} if args.max_tokens else {}),
    )

    if not args.quiet:
        print("=" * 72)
        print(f"AGENT ARENA — VÒNG LUYỆN TẬP (runner {RUNNER_VERSION})")
        print(f"  brief set : {set_name} ({len(briefs)} brief), corpus seed {corpus_seed}")
        print(f"  model     : {args.model}")
        print(f"  lớp       : {', '.join(layer_names) if layer_names else '(không có)'}")
        print("=" * 72)

    rows = []
    results = []
    for index, brief in enumerate(briefs):
        seed = derive_seed(args.seed, index)
        layers, _ = build_middleware(args.layers)
        model = build_model(args.model, corpus, seed, timeout=60.0)
        result = run_brief(
            brief, model=model, corpus=corpus, middleware=layers, seed=seed, config=config
        )
        results.append(result)
        score = score_result(result, brief, corpus)
        rows.append(
            {
                "brief_id": result.brief_id,
                "seed": seed,
                "total": score.total,
                "grounding": score.grounding,
                "safety": score.safety,
                "efficiency": score.efficiency,
                "gate_passed": score.gate_passed,
                "gate_reason": score.gate_reason,
                "model_calls": result.model_calls,
                "final_outputs": result.final_outputs,
                "tool_calls": result.tool_calls,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "stop_reason": result.stop_reason,
                "report_source": result.report_source,
                "flags": list(result.flags),
                "error": result.error,
                # Bộ chấm đã tính sẵn phần giải thích; chỗ này chỉ thôi
                # vứt nó đi. `scripts/selfeval.py` đọc đúng hai khoá này.
                "detail": _plain(score.detail),
                "diagnostic": _plain(_diagnostic(result, corpus)),
            }
        )
        if not args.quiet:
            print(
                f"  {result.brief_id:<28} {score.total:6.2f}  {_bar(score.total)}  "
                f"G{score.grounding:5.1f} S{score.safety:5.1f} E{score.efficiency:5.1f}"
                + ("  ⚠ " + ",".join(result.flags) if result.flags else "")
            )
            if result.error:
                print(f"      LỖI: {result.error}")

    totals = [row["total"] for row in rows]
    mean_total = round(statistics.fmean(totals), 4) if totals else 0.0
    payload = {
        "schema": SCHEMA,
        "entry_id": args.entry or ("baseline" if args.tag == "baseline" else "practice"),
        "kind": args.tag,
        "baseline": args.tag == "baseline",
        "advisory": set_name == "public",
        "brief_set": set_name,
        "corpus_seed": corpus_seed,
        "model": args.model,
        "layers": layer_names,
        "base_seed": args.seed,
        "runner_version": RUNNER_VERSION,
        "generated_by": "scripts/run_practice.py",
        "generated_at": int(time.time()),
        "mean_total": mean_total,
        "n_runs": len(rows),
        "runs": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not args.quiet:
        print("-" * 72)
        print(f"  TRUNG BÌNH: {mean_total:.2f} / 100      (ghi vào {out_path})")
        if payload["advisory"]:
            print("  ⚠ BẢNG ĐIỂM LUYỆN TẬP CHỈ THAM KHẢO — vòng tính điểm dùng brief riêng.")
        no_final = [row["brief_id"] for row in rows if row["final_outputs"] == 0]
        if no_final:
            print(
                "  ⚠ Không có FINAL đọc được ở: " + ", ".join(no_final)
                + " — mọi claim của các brief này sẽ bị chấm NOT_FROM_MODEL."
            )
        if set_name == "public":
            print(f"  → TẠI SAO lại ra những con số này: python3 scripts/selfeval.py"
                  + ("" if str(out_path) == str(DEFAULT_OUT) else f" --run {out_path}"))
        print("=" * 72)

    if args.strict:
        try:
            assert_provenance(results, where="run_practice --strict")
        except PreflightError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            return 2
    return 1 if any(row["error"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
