# rag-demo

A question-answering demo over company annual reports (PDF), following the approach in
[Ilya Rice: How I Won the Enterprise RAG Challenge](https://abdullin.com/ilya/how-to-build-best-rag/).

The knowledge base is 5 annual reports from round 2 of the
[Enterprise RAG Challenge](https://github.com/trustbit/enterprise-rag-challenge), with the organizers'
manually proofread gold answers for 17 questions. The winner's answers from
[IlyaRice/RAG-Challenge-2](https://github.com/IlyaRice/RAG-Challenge-2) (MIT) serve as a second reference.

## Roadmap

- [x] Project skeleton
- [x] Download dataset (5 annual reports + 17 questions with gold answers)
- [ ] Parsing: PDF -> per-page Markdown with Docling (keep page numbers and tables)
- [ ] Cleaning: inspect parse quality, regex cleanup of broken text
- [ ] Tagging: LLM-extracted document metadata, checked against `subset.csv`
- [ ] Ingestion: 300-token chunks (50 overlap), one FAISS index per company, page number in chunk metadata
- [ ] Retrieval: company routing -> vector search -> parent page -> LLM reranking
- [ ] Answering: prompt routing by answer type, structured output with CoT and page references
- [ ] Evaluation: compare against the gold answers and the winner's answers on 17 questions
- [ ] LangGraph workflow, web search fallback via MCP
- [ ] FastAPI service + Postman collection
- [ ] Dify version for comparison

Table serialization is left out on purpose: the original author found it slightly hurt results.
It may come back later as an optional experiment.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then add your OPENAI_API_KEY
```

## Data

```bash
uv run python -m rag_demo.download
```

Downloads into `data/erc2/` (git-ignored). Each source is pinned to a fixed commit, and each PDF
is named by its SHA-1 and verified after download.

The 5 companies have the most single-company questions among the 100 competition reports.
6 of the 17 gold answers are N/A, which tests whether the system makes up answers.
The 8 multi-company comparison questions are out of scope: none of them involve only these 5 companies.

| Company | Pages | Questions |
|---|---|---|
| AA Limited | 62 | 4 |
| HCA Healthcare, Inc. | 133 | 4 |
| Wheeler Real Estate Investment Trust, Inc. | 93 | 3 |
| Ritchie Bros. Auctioneers Incorporated | 144 | 3 |
| Albany International Corp. | 131 | 3 |

| File | Content |
|---|---|
| `pdf_reports/*.pdf` | 5 annual reports, 563 pages in total |
| `subset.csv` | Organizer metadata and tags (LLM-extracted by the organizers, used to generate questions) |
| `questions.json` | 17 questions with gold answers and gold page references (converted to 1-based) |
| `answers_ilya.json` | The winning system's answers to the same questions |
