# rag-demo

A question-answering demo over company annual reports (PDF), following the approach in
[Ilya Rice: How I Won the Enterprise RAG Challenge](https://abdullin.com/ilya/how-to-build-best-rag/).

The knowledge base is 5 annual reports from round 2 of the
[Enterprise RAG Challenge](https://github.com/trustbit/enterprise-rag-challenge), with the organizers'
manually proofread gold answers for 17 questions. The winner's answers from
[IlyaRice/RAG-Challenge-2](https://github.com/IlyaRice/RAG-Challenge-2) (MIT) serve as a second reference.

## Roadmap

- [x] Project skeleton
- [ ] Download dataset (5 annual reports + 17 questions with gold answers)
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
