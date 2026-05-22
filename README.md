# Smart Lost & Found

Smart Lost & Found is a backend application that matches lost items with found items using AI-generated descriptions and embedding similarity. Users can register lost and found items through either a FastAPI HTTP API or a command-line interface. The system analyzes uploaded item images, generates structured descriptions with a vision-language model, converts those descriptions into embeddings, and ranks possible matches using cosine similarity.

The project is built around the provided `ai/` package. We do not modify the public interface of that package. Instead, this project adds the software engineering layer around it: configuration management, validation, persistent storage, retries, concurrency, logging, testing, Docker deployment, and user-facing interfaces.

---

## Features

This project supports:

- FastAPI HTTP API for lost/found registration and matching
- Command-line interface with equivalent operations
- Provider-agnostic AI configuration through `.env`
- Offline mode for grading-safe execution without API keys
- Online mode with real AI providers such as OpenAI or Gemini
- PostgreSQL metadata storage through `asyncpg`
- Filesystem storage for uploaded images
- Embedding-based similarity matching
- Retries with exponential backoff for AI calls
- Bounded concurrency using `asyncio.Semaphore`
- Offline pytest test suite
- Dockerized deployment
- Demo artefact generation under `artefacts/`

---

## Project Structure

```text
.
├── ai/                     # Provided AI module; treated as external dependency
├── artefacts/              # Demo output files
├── data/                   # Provided sample lost/found item images
├── docs/                   # Architecture documentation
├── report/                 # Report and presentation files
├── scripts/                # Demo and benchmark scripts
├── src/
│   ├── api.py              # FastAPI HTTP API
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Typed environment configuration
│   ├── models.py           # Pydantic application models
│   ├── concurrency/        # Bounded async pipeline
│   ├── core/               # Matching logic and validation
│   ├── services/           # AI service wrapper
│   └── storage/            # Repository implementations
├── storage/                # Runtime image storage
├── tests/                  # Pytest test suite
├── Dockerfile
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Environment Configuration

The application reads configuration from environment variables. For local development, copy the example environment file and fill in real values.

```bash
cp .env.example .env
```

Example `.env` for OpenAI:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key_here

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

LOG_LEVEL=INFO
DATABASE_URL=postgresql+asyncpg://postgres:dev@localhost:55432/lostfound
IMAGE_STORAGE_DIR=./storage/images
MAX_IMAGE_SIZE_MB=5
HTTP_PORT=8000
```

`.env.example` is safe to commit because it contains placeholders only. Real secrets must stay in `.env` or `.env.docker`, and those files should not be committed.

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install pinned dependencies:

```bash
python3 -m pip install -r requirements.txt
```

This installs the backend framework, AI provider SDKs, database driver, testing tools, and runtime dependencies required by the application.

---

## PostgreSQL Setup

The project stores item metadata in PostgreSQL. A local PostgreSQL instance can be started with Docker:

```bash
docker run -d --name pg \
  -e POSTGRES_PASSWORD=dev \
  -e POSTGRES_DB=lostfound \
  -p 55432:5432 \
  postgres:16
```

Check that the container is running:

```bash
docker ps
```

Use this database URL for local runs:

```env
DATABASE_URL=postgresql+asyncpg://postgres:dev@localhost:55432/lostfound
```

Metadata such as item ID, status, user text, AI description, embedding bytes, image path, and timestamp is stored in PostgreSQL. Uploaded image files are stored separately in the filesystem under `storage/images/`.

---

## Offline Demo

Offline mode runs without API keys or network access. It uses deterministic fake providers to simulate VLM descriptions and embeddings.

```bash
python3 demo_ai.py --offline
```

This is useful for grading-safe execution and for confirming that the AI pipeline shape still works without consuming provider quota.

The demo also writes its result to:

```text
artefacts/demo_results.json
```

---

## Online Demo

Online mode uses the real provider configured in `.env`.

Load the environment variables:

```bash
set -a
source .env
set +a
```

Run the demo:

```bash
python3 demo_ai.py
```

This sends sample images from `data/lost/` and `data/found/` to the configured provider, generates descriptions and embeddings, and prints the top matches.

The same result is also written to:

```text
artefacts/demo_results.json
```

---

## HTTP API

Start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

The server runs locally on:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","version":"1.0.0"}
```

The API is the main web-facing interface of the system. It allows clients to register items, list stored items, and retrieve match results.

---

## Register a Lost Item

```bash
curl -X POST http://127.0.0.1:8000/items/lost \
  -F "user_text=lost backpack" \
  -F "image=@data/lost/backpack_navy.png"
```

This endpoint validates the image, stores the image file, generates a VLM description, creates an embedding, and saves the item metadata.

---

## Register a Found Item

```bash
curl -X POST http://127.0.0.1:8000/items/found \
  -F "user_text=found backpack" \
  -F "image=@data/found/backpack_navy_2.png"
```

Found items go through the same processing pipeline as lost items. They are stored with status `found` and can later be compared against lost items.

---

## Retrieve Matches

Replace `<LOST_ITEM_ID>` with the ID returned from the lost item registration response.

```bash
curl "http://127.0.0.1:8000/items/<LOST_ITEM_ID>/matches?k=3"
```

The API compares the selected item against the opposite pool and returns the top-k most similar items with similarity scores.

---

## List Items

List all items:

```bash
curl "http://127.0.0.1:8000/items"
```

List only lost items:

```bash
curl "http://127.0.0.1:8000/items?status=lost"
```

List only found items:

```bash
curl "http://127.0.0.1:8000/items?status=found"
```

This is useful for checking which items were registered and whether metadata persistence is working correctly.

---

## CLI Usage

Show available CLI commands:

```bash
python3 -m src.cli --help
```

The CLI mirrors the HTTP API and is useful for running the system without starting the web server.

Register a lost item:

```bash
python3 -m src.cli register-lost \
  --image data/lost/backpack_navy.png \
  --text "lost backpack"
```

Register a found item:

```bash
python3 -m src.cli register-found \
  --image data/found/backpack_navy_2.png \
  --text "found backpack"
```

Search matches:

```bash
python3 -m src.cli search-matches \
  --id <ITEM_ID> \
  --k 3
```

List items:

```bash
python3 -m src.cli list --status lost
```

The CLI uses the same core matching logic as the API, so both interfaces remain consistent.

---

## Testing

Run the full test suite:

```bash
PYTHONPATH=. pytest
```

Current result:

```text
91 passed
```

Run coverage:

```bash
PYTHONPATH=. pytest --cov=src --cov-report=term-missing
```

Current coverage:

```text
76%
```

Run only the provided AI smoke tests:

```bash
pytest tests/test_ai_smoke.py
```

All tests run offline. Fake VLM and embedding providers are used so the test suite does not require real API keys, network access, or provider quota.

---

## Type Checking

A type checker was run as part of the validation process:

```bash
mypy src
```

The result should be documented in the report. Some warnings may come from optional third-party packages or the provided `ai/` package, which is treated as an external dependency.

---

## Concurrency Benchmark

The project includes a benchmark comparing sequential execution with bounded concurrent execution.

Run:

```bash
python3 scripts/benchmark.py
```

Example output:

```text
Benchmark: sequential vs concurrent
Items: 20
Sequential time: 4.00s
Concurrent time: 0.80s
Speedup: 5.00x
Results equal: True
```

The benchmark demonstrates that I/O-bound operations can be processed faster with bounded concurrency. The implementation uses `asyncio.gather` and `asyncio.Semaphore`.

---

## Docker Deployment

Build the Docker image:

```bash
docker build --no-cache -t lost-found-ai .
```

For Docker runs, create `.env.docker`. Since the app runs inside a container, it must reach PostgreSQL through `host.docker.internal`:

```env
DATABASE_URL=postgresql+asyncpg://postgres:dev@host.docker.internal:55432/lostfound
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
LOG_LEVEL=INFO
IMAGE_STORAGE_DIR=./storage/images
MAX_IMAGE_SIZE_MB=5
HTTP_PORT=8000
```

Run the container:

```bash
docker run --env-file .env.docker -p 8000:8000 lost-found-ai
```

Check the API inside Docker:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","version":"1.0.0"}
```

The Dockerfile exposes port `8000` and starts the FastAPI server using Uvicorn.

---

## Storage Design

The project separates metadata storage from image storage.

Metadata is stored in PostgreSQL and includes:

```text
item id
status
user text
AI description
embedding bytes
image path
timestamp
```

Image files are stored in:

```text
storage/images/
```

This design keeps large image files outside the database while keeping searchable metadata in PostgreSQL.

---

## Robustness Features

The project includes several robustness mechanisms:

- explicit input validation
- image MIME/type validation
- maximum image size enforcement
- corrupted image rejection
- retries with exponential backoff
- timeout handling
- embedding cache
- structured logging
- clean API error responses
- bounded concurrency
- PostgreSQL fallback behavior during development

The code avoids:

```text
bare except:
except Exception: pass
hardcoded API keys
```

---

## Provider-Agnostic Configuration

The system is provider-agnostic. Provider selection happens through `.env`.

OpenAI example:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

Gemini example:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash-lite
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001
```

Switching providers does not require changing application code.

---

## Offline vs Online Modes

Offline mode:

```bash
python3 demo_ai.py --offline
```

This mode is deterministic and does not call external providers.

Online mode:

```bash
python3 demo_ai.py
```

This mode uses the real provider configured through `.env`.

Both modes write demo output into:

```text
artefacts/demo_results.json
```

---

## Demo Artefacts

Each demo run writes a JSON output file:

```text
artefacts/demo_results.json
```

The artefact records:

```text
run timestamp
execution mode
query item
query text
top matches
similarity scores
```

This makes demo runs reproducible and provides evidence for the report.

---

## Cleaning Local Files

Generated local files should not be committed:

```text
.env
.env.docker
storage/
lostfound.json
.pytest_cache/
.mypy_cache/
.coverage
__pycache__/
*.pyc
.DS_Store
```

These should be excluded through `.gitignore` and `.dockerignore`.
