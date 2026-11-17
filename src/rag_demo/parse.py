"""Parse annual report PDFs into per-page Markdown with Docling.

For each PDF two files are written:
    parsed/<sha1>.docling.json   full Docling document (layout, tables, metadata),
                                 kept so later steps can re-export without re-parsing
    pages/<sha1>.json            {"sha1", "page_count", "pages": [{"page", "text"}]}
                                 with 1-based page numbers and Markdown text per page

OCR is off by default, following the original approach: parse the text layer first,
and only OCR documents whose text turns out to be broken (see the cleaning step).

Usage:
    uv run python -m rag_demo.parse              # all reports
    uv run python -m rag_demo.parse aa781901     # only reports whose sha1 starts with this
    uv run python -m rag_demo.parse --ocr ...    # force OCR
"""

import argparse
import json
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from rag_demo.download import DATA_DIR

PDF_DIR = DATA_DIR / "pdf_reports"
PARSED_DIR = DATA_DIR / "parsed"
PAGES_DIR = DATA_DIR / "pages"


def build_converter(ocr: bool) -> DocumentConverter:
    options = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def parse_pdf(converter: DocumentConverter, pdf: Path) -> dict:
    doc = converter.convert(pdf).document
    doc.save_as_json(PARSED_DIR / f"{pdf.stem}.docling.json")

    page_count = len(doc.pages)
    pages = [
        {"page": n, "text": doc.export_to_markdown(page_no=n)}
        for n in range(1, page_count + 1)
    ]
    result = {"sha1": pdf.stem, "page_count": page_count, "pages": pages}
    (PAGES_DIR / f"{pdf.stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prefixes", nargs="*", help="sha1 prefixes to parse (default: all)")
    parser.add_argument("--ocr", action="store_true", help="enable OCR")
    args = parser.parse_args()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.prefixes:
        pdfs = [p for p in pdfs if p.stem.startswith(tuple(args.prefixes))]
    if not pdfs:
        raise SystemExit(f"no PDFs found in {PDF_DIR}; run rag_demo.download first")

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    converter = build_converter(args.ocr)

    for pdf in pdfs:
        start = time.perf_counter()
        result = parse_pdf(converter, pdf)
        chars = sum(len(p["text"]) for p in result["pages"])
        empty = sum(1 for p in result["pages"] if not p["text"].strip())
        print(
            f"{pdf.stem[:8]}  {result['page_count']:>4} pages  {chars:>8} chars  "
            f"{empty:>3} empty pages  {time.perf_counter() - start:6.1f}s"
        )


if __name__ == "__main__":
    main()
