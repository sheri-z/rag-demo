# Temporary public link

How to give other people a link to the API, using a Cloudflare quick tunnel. The API keeps
running on your own computer; the tunnel forwards public HTTPS traffic to `127.0.0.1:8000`.

```
browser / Postman -> https://<random>.trycloudflare.com -> cloudflared (your computer) -> FastAPI -> LangGraph
```

> **Do this on a personal computer only.** A tunnel opens a path from the internet to the
> machine it runs on, through any firewall in between. On a company computer or network this
> usually breaks the security policy, and security software may flag it.

The link works only while both the API and the tunnel are running, and it changes every time the
tunnel restarts. Quick tunnels need no Cloudflare account and are meant for testing and demos.

## 1. Set up the project

Requires git, [uv](https://docs.astral.sh/uv/) and about 3 GB of disk space (Docling and its models).

```bash
git clone git@github.com:sheri-z/rag-demo.git
cd rag-demo
uv sync
cp .env.example .env
```

Edit `.env`:

- `OPENAI_API_KEY`: your OpenAI key.
- `API_KEY`: a new random key for the HTTP API, for example from
  `python3 -c 'import secrets; print(secrets.token_urlsafe(24))'`.

## 2. Build the knowledge base

The data is not in the repository, so build it once. This takes about 15 minutes on a recent Mac
(most of it PDF parsing, plus Docling's model download on the first run) and about $0.02 of OpenAI usage.

```bash
uv run python -m rag_demo.download
uv run python -m rag_demo.parse
uv run python -m rag_demo.clean
uv run python -m rag_demo.tag
uv run python -m rag_demo.ingest
```

Check that it works:

```bash
uv run python -m rag_demo.workflow "What was the Dividend per share (in USD) for Ritchie Bros. Auctioneers Incorporated?"
```

The answer should be `0.27`, from the knowledge base, citing page 38.

## 3. Limit what the link can cost

Every `/ask` call costs about $0.005-0.01 of OpenAI usage, and a link can be forwarded or found by scanners.

- In the OpenAI dashboard, set a monthly budget limit for the project or key.
- Keep the API limits in `.env` low enough for your budget: `RATE_LIMIT_PER_MINUTE` (default 10 per
  client IP) and `DAILY_REQUEST_LIMIT` (default 200 in total).
- Give the `API_KEY` only to the people who should use the link. Never share `OPENAI_API_KEY`.

## 4. Start the API and the tunnel

Install `cloudflared` once:

```bash
brew install cloudflared            # macOS
# winget install --id Cloudflare.cloudflared   (Windows)
```

Then, in two terminals:

```bash
# terminal 1
uv run python -m rag_demo.api

# terminal 2
cloudflared tunnel --url http://127.0.0.1:8000
```

`cloudflared` prints the public URL, for example `https://example-words.trycloudflare.com`.

Test it from any network:

```bash
curl https://<your-url>.trycloudflare.com/health
```

## 5. Share it

Send people:

- the Swagger UI, `https://<your-url>.trycloudflare.com/docs`: click **Authorize**, enter the key,
  then try `POST /ask` in the browser.
- or the Postman collection `postman/rag-demo.postman_collection.json`, with `base_url` set to the
  tunnel URL and `api_key` set to the key.
- the `API_KEY`, sent separately from the link.

## 6. Stop it

Press Ctrl+C in both terminals. The link stops working right away. If the key was shared widely,
put a new `API_KEY` in `.env` before the next demo.
