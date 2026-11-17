"""Download 5 annual reports from the Enterprise RAG Challenge round 2 with gold answers.

The 5 companies were picked from the 100 competition reports because they have the most
single-company questions (17 in total, mixed answer types, 6 with gold answer N/A).

Sources, each pinned to a fixed commit so the data never changes under us:
    trustbit/enterprise-rag-challenge   PDFs and manually proofread gold answers (organizers)
    IlyaRice/RAG-Challenge-2            company metadata (subset.csv) and the winner's answers

Output in data/erc2/:
    pdf_reports/<sha1>.pdf     the 5 reports, verified against the SHA-1 in the file name
    subset.csv                 organizer metadata and tags for the 5 companies
    questions.json             17 questions with gold answers and gold page references
    answers_ilya.json          the winning system's answers to the same questions

Usage:
    uv run python -m rag_demo.download
"""

import csv
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

ORGANIZER_URL = (
    "https://raw.githubusercontent.com/trustbit/enterprise-rag-challenge/"
    "a4abdb9478c7c1021a916006d56605cc94bd4dde/round2"
)
ILYA_URL = (
    "https://raw.githubusercontent.com/IlyaRice/RAG-Challenge-2/"
    "452d688d1e0d6c3dc5150f9e42c2facb477178b7/data/erc2_set"
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "erc2"
PDF_DIR = DATA_DIR / "pdf_reports"

COMPANIES = {
    "aa781901e117281bfee6f8e4bea6fc9c9bada62e": "AA Limited",
    "a69ebee82a9acd54117407ca3697db0d7bfbfdac": "HCA Healthcare, Inc.",
    "b947c33b370d8a3251ef9c36ce7d71e8d16f4f8e": "Wheeler Real Estate Investment Trust, Inc.",
    "78c71282723c2d66216cbba13183d19349d302b8": "Ritchie Bros. Auctioneers Incorporated",
    "da663e46fbf02ec8a90b3f3c1079ef4c9f7907e1": "Albany International Corp.",
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def company_of(question: str) -> str | None:
    hits = [(sha, name) for sha, name in COMPANIES.items() if name in question]
    return hits[0][0] if len(hits) == 1 else None


def parse_reference(ref: str) -> dict:
    """Gold references look like "<sha1>:<page_index>" with 0-based page indexes.

    Converted to 1-based page numbers to match the rest of the pipeline. Checked by hand:
    e.g. Albany's R&D spending (31.4) appears on page 16, gold reference index 15.
    """
    sha, page_index = ref.split(":")
    return {"sha1": sha, "page": int(page_index) + 1}


def download_pdfs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for digest, name in COMPANIES.items():
        dest = PDF_DIR / f"{digest}.pdf"
        if dest.exists() and sha1(dest.read_bytes()) == digest:
            print(f"skip  {dest.name}  {name}")
            continue
        data = fetch(f"{ORGANIZER_URL}/pdfs/{dest.name}")
        if sha1(data) != digest:
            sys.exit(f"checksum mismatch for {dest.name}")
        dest.write_bytes(data)
        print(f"ok    {dest.name}  {name} ({len(data) / 1e6:.1f} MB)")


def write_subset() -> None:
    rows = list(csv.DictReader(io.StringIO(fetch(f"{ILYA_URL}/subset.csv").decode())))
    rows = [r for r in rows if r["sha1"] in COMPANIES]
    with open(DATA_DIR / "subset.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"ok    subset.csv ({len(rows)} companies)")


def write_questions() -> list[dict]:
    gold = json.loads(fetch(f"{ORGANIZER_URL}/answers.json"))
    questions = []
    for text, info in gold.items():
        sha = company_of(text)
        if sha is None:
            continue
        questions.append(
            {
                "text": text,
                "kind": info["kind"],
                "sha1": sha,
                "company": COMPANIES[sha],
                # several answers / reference pools mean any one of them is accepted
                "gold_answers": info["answers"],
                "gold_reference_pools": [
                    [parse_reference(r) for r in pool] for pool in info["reference_pools"]
                ],
            }
        )
    (DATA_DIR / "questions.json").write_text(json.dumps(questions, indent=2, ensure_ascii=False))
    print(f"ok    questions.json ({len(questions)} questions)")
    return questions


def write_ilya_answers(questions: list[dict]) -> None:
    ours = {q["text"] for q in questions}
    answers = json.loads(fetch(f"{ILYA_URL}/answers_1st_place_o3-mini.json"))["answers"]
    answers = [a for a in answers if a["question_text"] in ours]
    (DATA_DIR / "answers_ilya.json").write_text(json.dumps(answers, indent=2, ensure_ascii=False))
    print(f"ok    answers_ilya.json ({len(answers)} answers)")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_pdfs()
    write_subset()
    questions = write_questions()
    write_ilya_answers(questions)
    print(f"done -> {DATA_DIR}")


if __name__ == "__main__":
    main()
