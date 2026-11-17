"""Score the system on the 17 questions with the organizers' rules, next to the winner's answers.

Scoring follows the organizers' rank.py (trustbit/enterprise-rag-challenge):
    value score (0 to 1)
        number   within 1% of a gold value
        boolean  / name   case-insensitive equality
        names    Jaccard overlap of the name sets
        N/A only matches N/A
    reference score (0 to 1), starting from 1
        -0.1  for each cited page that is not a gold page
        -0.25 for each gold reference pool with no cited page
    total = value + reference / 2, so at most 1.5 per question

Each run is saved to data/erc2/runs/<name>.json with the settings used, so different
configurations can be compared later.

Usage:
    uv run python -m rag_demo.evaluate                 # saved as runs/<timestamp>.json
    uv run python -m rag_demo.evaluate --name baseline
    uv run python -m rag_demo.evaluate --name baseline --repeat 3   # mean, min, max
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from rag_demo import retrieve
from rag_demo.answer import answer
from rag_demo.config import get_settings
from rag_demo.paths import DATA_DIR

RUNS_DIR = DATA_DIR / "runs"
WORKERS = 4


def compare(kind: str, gold: str, predicted) -> float:
    """Value score for one gold answer, same rules as the organizers' compare()."""
    if predicted == "N/A" and gold == "N/A":
        return 1.0
    if gold == "N/A" or predicted == "N/A":
        return 0.0
    if kind == "number":
        try:
            gold_f, predicted_f = float(gold), float(predicted)
        except (TypeError, ValueError):
            return 0.0
        return 1.0 if abs(predicted_f - gold_f) < 0.01 * gold_f else 0.0
    if kind in ("boolean", "name"):
        return 1.0 if str(gold).strip().lower() == str(predicted).strip().lower() else 0.0
    if kind == "names":
        if isinstance(predicted, str):
            predicted = [p.strip() for p in predicted.split(",")]
        gold_names = set(str(gold).strip().lower().split(","))
        predicted_names = {str(p).strip().lower() for p in predicted}
        return len(gold_names & predicted_names) / len(gold_names | predicted_names)
    raise ValueError(f"unknown kind {kind}")


def reference_score(cited: list[str], gold_pools: list[list[str]]) -> float:
    """References are "<sha1>:<page>" strings."""
    if not gold_pools and not cited:
        return 1.0
    gold = {ref for pool in gold_pools for ref in pool}
    score = 1.0
    score -= 0.1 * sum(1 for ref in cited if ref not in gold)
    score -= 0.25 * sum(1 for pool in gold_pools if not set(cited) & set(pool))
    return max(0.0, score)


def score_answer(question: dict, value, cited: list[str]) -> dict:
    gold_pools = [
        [f"{r['sha1']}:{r['page']}" for r in pool] for pool in question["gold_reference_pools"]
    ]
    value_score = max(compare(question["kind"], g, value) for g in question["gold_answers"])
    ref_score = reference_score(cited, gold_pools)
    return {"value": value_score, "reference": ref_score, "total": value_score + ref_score / 2}


def ilya_scores(questions: list[dict]) -> list[dict]:
    answers = {
        a["question_text"]: a for a in json.loads((DATA_DIR / "answers_ilya.json").read_text())
    }
    results = []
    for q in questions:
        a = answers[q["text"]]
        cited = [f"{r['pdf_sha1']}:{r['page_index'] + 1}" for r in a["references"]]  # 0-based
        results.append(
            {"value": a["value"], "cited": cited, "score": score_answer(q, a["value"], cited)}
        )
    return results


def run(questions: list[dict]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        answers = list(pool.map(lambda q: answer(q["text"], q["kind"]), questions))
    results = []
    for q, a in zip(questions, answers):
        cited = [f"{r['sha1']}:{r['page']}" for r in a.references]
        results.append({**a.to_dict(), "cited": cited, "score": score_answer(q, a.value, cited)})
    return results


def totals(results: list[dict]) -> dict:
    n = len(results)
    value = sum(r["score"]["value"] for r in results)
    reference = sum(r["score"]["reference"] for r in results)
    return {
        "questions": n,
        "value": round(value, 2),
        "reference": round(reference, 2),
        "total": round(value + reference / 2, 2),
        "max_total": 1.5 * n,
    }


def settings_snapshot() -> dict:
    s = get_settings()
    return {
        "llm_model": s.llm_model,
        "answer_model": s.answer_model,
        "embedding_model": s.embedding_model,
        "sample_chunks": retrieve.SAMPLE_CHUNKS,
        "top_pages": retrieve.TOP_PAGES,
        "rerank_batch": retrieve.RERANK_BATCH,
        "llm_weight": retrieve.LLM_WEIGHT,
    }


def short(value, width: int = 26) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def print_report(questions: list[dict], ours: list[dict], ilya: list[dict]) -> None:
    print(
        f"{'kind':8} {'gold':26} {'ours':26} {'val':>4} {'ref':>4}   "
        f"{'ilya':26} {'val':>4} {'ref':>4}"
    )
    for q, o, i in zip(questions, ours, ilya):
        gold = q["gold_answers"][0] if len(q["gold_answers"]) == 1 else q["gold_answers"]
        print(
            f"{q['kind']:8} {short(gold):26} {short(o['value']):26} "
            f"{o['score']['value']:4.1f} {o['score']['reference']:4.2f}   "
            f"{short(i['value']):26} {i['score']['value']:4.1f} {i['score']['reference']:4.2f}"
        )
    print()
    for name, results in (("ours", ours), ("ilya", ilya)):
        t = totals(results)
        print(
            f"{name}: value {t['value']}/{t['questions']}  "
            f"reference {t['reference']}/{t['questions']}"
            f"  total {t['total']}/{t['max_total']}"
        )


def save_run(name: str, questions: list[dict], ours: list[dict], ilya: list[dict], elapsed: float):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "settings": settings_snapshot(),
        "seconds": round(elapsed),
        "totals": totals(ours),
        "ilya_totals": totals(ilya),
        "answers": [
            {"question": q["text"], "kind": q["kind"], "gold": q["gold_answers"], **o}
            for q, o in zip(questions, ours)
        ],
    }
    path = RUNS_DIR / f"{name}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument(
        "--repeat", type=int, default=1, help="runs to average; answers vary between runs"
    )
    args = parser.parse_args()

    questions = json.loads((DATA_DIR / "questions.json").read_text())
    ilya = ilya_scores(questions)
    run_totals = []
    for i in range(1, args.repeat + 1):
        name = args.name if args.repeat == 1 else f"{args.name}-{i}"
        start = time.perf_counter()
        ours = run(questions)
        elapsed = time.perf_counter() - start
        print(f"\n=== {name} ===")
        print_report(questions, ours, ilya)
        path = save_run(name, questions, ours, ilya, elapsed)
        print(f"{elapsed:.0f}s  -> {path}")
        run_totals.append(totals(ours))

    if args.repeat > 1:
        print(f"\n=== {args.name}: {args.repeat} runs ===")
        for key in ("value", "reference", "total"):
            values = [t[key] for t in run_totals]
            print(
                f"{key:9} mean {sum(values) / len(values):5.2f}  "
                f"min {min(values):5.2f}  max {max(values):5.2f}"
            )


if __name__ == "__main__":
    main()
