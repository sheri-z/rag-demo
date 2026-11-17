"""Clean parsed reports: re-export compact Markdown per page and check text quality.

Reads parsed/<sha1>.docling.json (no re-parsing) and writes pages_clean/<sha1>.json
in the same format as pages/: {"sha1", "page_count", "pages": [{"page", "text"}]}.

Most noise is fixed by Docling export options:
    compact_tables      drop the column-alignment padding (16-52% of characters)
    escape_html=False   keep "&" instead of "&amp;"
    escape_underscores  keep "_" instead of "\\_"
    image_placeholder   drop "<!-- image -->" lines (we do not use images)
The few regexes below handle what export options cannot.

It also flags documents with a broken text layer (glyph codes, garbled words).
Those should be re-parsed with `rag_demo.parse --ocr`, as in the original approach.

Usage:
    uv run python -m rag_demo.clean
"""

import json
import re

from docling_core.types.doc import DoclingDocument

from rag_demo.paths import CLEAN_DIR, PAGES_DIR, PARSED_DIR

EXPORT_OPTIONS = dict(
    compact_tables=True,
    escape_html=False,
    escape_underscores=False,
    image_placeholder="",
)

# (pattern, replacement) applied in order
REGEX_RULES = [
    (re.compile(r"\s*\.{4,}\s*"), " "),  # table-of-contents dot leaders
    (re.compile(r"[ \t]+\n"), "\n"),  # trailing spaces
    (re.compile(r"\n{3,}"), "\n\n"),  # extra blank lines left by removed images
]

# Signs of a broken text layer
GLYPH_ARTIFACTS = re.compile(r"GLYPH<|/uni[0-9A-F]{4}|\(cid:\d+\)")
WORD = re.compile(r"[A-Za-z]{3,}")
VOWEL = re.compile(r"[aeiouyAEIOUY]")
MAX_VOWELLESS_SHARE = 0.05  # normal English text is well under 2%


def clean_text(text: str) -> str:
    for pattern, repl in REGEX_RULES:
        text = pattern.sub(repl, text)
    return text.strip()


def quality(text: str) -> dict:
    words = WORD.findall(text)
    vowelless = sum(1 for w in words if not VOWEL.search(w))
    return {
        "glyph_artifacts": len(GLYPH_ARTIFACTS.findall(text)),
        "vowelless_share": vowelless / max(len(words), 1),
    }


def clean_document(path) -> dict:
    doc = DoclingDocument.load_from_json(path)
    sha1 = path.name.removesuffix(".docling.json")
    pages = [
        {"page": n, "text": clean_text(doc.export_to_markdown(page_no=n, **EXPORT_OPTIONS))}
        for n in range(1, len(doc.pages) + 1)
    ]
    return {"sha1": sha1, "page_count": len(pages), "pages": pages}


def main() -> None:
    paths = sorted(PARSED_DIR.glob("*.docling.json"))
    if not paths:
        raise SystemExit(f"nothing in {PARSED_DIR}; run rag_demo.parse first")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    for path in paths:
        result = clean_document(path)
        sha1 = result["sha1"]
        (CLEAN_DIR / f"{sha1}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

        raw = json.loads((PAGES_DIR / f"{sha1}.json").read_text())
        before = sum(len(p["text"]) for p in raw["pages"])
        text = "\n".join(p["text"] for p in result["pages"])
        q = quality(text)
        broken = q["glyph_artifacts"] > 0 or q["vowelless_share"] > MAX_VOWELLESS_SHARE
        shrink = 1 - len(text) / before
        print(
            f"{sha1[:8]}  {before:>7} -> {len(text):>7} chars ({shrink:4.0%} smaller)"
            f"  glyphs={q['glyph_artifacts']}  vowelless={q['vowelless_share']:.1%}"
            f"  {'BROKEN: re-parse with --ocr' if broken else 'ok'}"
        )


if __name__ == "__main__":
    main()
