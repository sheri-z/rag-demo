"""Data locations, in one module with no heavy imports.

Pipeline steps import paths from here instead of from each other. Importing
rag_demo.parse just for a path would load Docling and torch, and torch's OpenMP
runtime crashes FAISS searches in the same process on macOS.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "erc2"

PDF_DIR = DATA_DIR / "pdf_reports"  # download
PARSED_DIR = DATA_DIR / "parsed"  # parse: full Docling JSON
PAGES_DIR = DATA_DIR / "pages"  # parse: raw per-page Markdown
CLEAN_DIR = DATA_DIR / "pages_clean"  # clean
METADATA_PATH = DATA_DIR / "metadata.json"  # tag
INDEX_DIR = DATA_DIR / "indexes"  # ingest
