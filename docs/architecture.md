# Smart Lost & Found Architecture

## 1. System Overview

Smart Lost & Found is a backend application designed to help users find possible matches between lost items and found items using AI-generated descriptions and embedding similarity. Each item is registered with an image and an optional user-provided text description. The system analyzes the uploaded image, generates a structured item description, converts that description into an embedding vector, and compares it against items from the opposite pool to determine the most likely matches.

The project is built around the provided `ai/` package, which is treated as an external dependency and is not modified. Our work focuses on building the software engineering layer around this package, including configuration management, validation, persistent storage, API and CLI interfaces, retries, concurrency, logging, testing, and Docker deployment. The architecture separates responsibilities into clear layers. The AI module handles only AI-specific operations, while the rest of the application manages validation, orchestration, storage, concurrency, and deployment concerns.

---

## 2. High-Level Data Flow

When a user registers an item, the application follows the following workflow:

1. The user submits an image and optional text through either the HTTP API or the CLI.
2. The input is validated before any expensive operation occurs.
3. The uploaded image is checked for supported format, size limit, and corruption.
4. The image file is copied into the configured filesystem storage directory.
5. The AI service sends the image and text to the configured vision-language provider.
6. The provider returns a structured description of the item.
7. The description is converted into searchable text.
8. The embedding provider converts the searchable text into an embedding vector.
9. The item metadata, AI description, embedding, timestamp, and image path are stored.
10. When a match request is made, the system compares the query item embedding against embeddings from the opposite item pool using cosine similarity.
11. The top-k most similar items are returned together with similarity scores and explanation text.

This architecture keeps the workflow modular and testable because each stage of processing is handled by a separate component.

---

## 3. Main Components

### 3.1 Configuration Layer

The configuration layer is implemented in:

```text
src/config.py
```

This layer uses `pydantic-settings` to load all runtime settings from `.env` files and environment variables. Important configuration values include provider selection, API keys, database URL, image storage directory, maximum image size, log level, and HTTP server configuration.

Examples of loaded settings include:

```text
LLM_PROVIDER
LLM_MODEL
OPENAI_API_KEY
GOOGLE_API_KEY
ANTHROPIC_API_KEY
EMBEDDING_PROVIDER
EMBEDDING_MODEL
DATABASE_URL
IMAGE_STORAGE_DIR
MAX_IMAGE_SIZE_MB
LOG_LEVEL
HTTP_PORT
```

This design prevents secrets from being hardcoded in the codebase and allows providers to be switched without changing application logic.

---

### 3.2 HTTP API Layer

The HTTP API is implemented in:

```text
src/api.py
```

The application uses FastAPI and exposes the required Topic 1 endpoints:

```text
POST /items/lost
POST /items/found
GET /items/{id}/matches?k=N
GET /items?status=...
```

A health endpoint is also provided:

```text
GET /health
```

The API layer is intentionally thin. It validates request structure, delegates processing to the matching service, and returns structured JSON responses. Business logic is not placed directly inside endpoint handlers.

The API also defines explicit exception handlers for validation failures, missing items, AI provider failures, and generic application errors. This ensures that users receive clean JSON responses instead of raw Python tracebacks.

---

### 3.3 CLI Layer

The command-line interface is implemented in:

```text
src/cli.py
```

The CLI mirrors the same operations supported by the HTTP API:

```text
register-lost
register-found
search-matches
list
```

The CLI allows the application to be used without running the FastAPI server. The CLI uses the same services as the API, ensuring consistent behavior across both interfaces.

---

### 3.4 AI Service Layer

The AI wrapper layer is implemented in:

```text
src/services/ai_service.py
```

This service wraps the provided `ai/` package and adds production-oriented behavior around it.

Responsibilities include:

```text
calling the VLM provider
calling the embedding provider
retrying transient failures
exponential backoff
timeout handling
bounded concurrency
embedding caching
structured logging
provider-independent execution
```

The wrapper does not modify the public interface of the provided AI module. Instead, it composes around the module and adds reliability features externally.

Embedding caching is implemented inside this service. If the same text is embedded multiple times during a session, the cached embedding vector is reused instead of calling the provider again.

---

### 3.5 Core Matching Layer

The main business logic is implemented in:

```text
src/core/matching_service.py
```

This service coordinates the registration and matching process. It receives its dependencies through composition:

```text
repository
ai_service
settings
```

This makes the service easier to test because fake repositories and fake AI providers can be injected during offline tests.

Responsibilities of the matching layer include:

```text
validating registration requests
storing uploaded image files
requesting AI descriptions
requesting embeddings
saving metadata
retrieving opposite-pool candidates
computing similarity scores
returning top-k matches
```

The matching process compares lost items against found items and found items against lost items using cosine similarity over embedding vectors.

---

### 3.6 Validation Layer

Validation logic is implemented in:

```text
src/core/validation.py
```

This layer validates user-facing input before the system proceeds to AI or database operations.

Validation checks include:

```text
image existence
supported MIME type
maximum image size
corrupted image detection
user text validation
```

Invalid input is rejected with clear application-level errors that are later converted into clean API or CLI responses.

---

### 3.7 Storage Layer

Storage logic is implemented in:

```text
src/storage/repository.py
```

The architecture separates metadata persistence from image blob persistence.

Metadata is stored in PostgreSQL through `asyncpg`. Stored metadata includes:

```text
item id
status
user text
image path
AI description
embedding bytes
timestamp
```

Uploaded images are stored separately in the filesystem under:

```text
storage/images/
```

This design keeps the database focused on searchable metadata while the filesystem stores larger binary image files.

The repository layer also supports fallback storage implementations such as JSON and in-memory repositories for development and testing.

---

### 3.8 Concurrency Layer

Concurrency support is implemented in:

```text
src/concurrency/pipeline.py
```

This layer uses:

```text
asyncio.gather
asyncio.Semaphore
```

The semaphore bounds parallelism and prevents too many simultaneous provider calls from being executed at once. This helps protect against provider rate limits and excessive resource usage.

The concurrency pipeline preserves input order and isolates failures so that one failed task does not necessarily terminate the entire batch.

A separate benchmark script compares sequential execution against concurrent execution and demonstrates the performance improvement achieved through asynchronous processing.

---

## 4. Provider-Agnostic Design

The system is designed to remain provider-independent. Provider selection is controlled entirely through environment variables.

Example configuration:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

The same code can run with OpenAI, Gemini, or Anthropic by changing only configuration values.

The project supports two runtime modes:

```text
offline mode
online mode
```

Offline mode uses deterministic fake providers and avoids all network calls. Online mode uses real providers selected through `.env`.

---

## 5. Persistence Design

### 5.1 Metadata Persistence

Metadata is stored in PostgreSQL. This allows the application to persist registered items across restarts and efficiently retrieve them later.

The repository pattern hides database implementation details from the rest of the application. The matching service interacts only with repository interfaces rather than directly accessing SQL logic.

---

### 5.2 Image Persistence

Uploaded image files are stored on disk rather than inside the database.

The configured image directory is read from:

```text
IMAGE_STORAGE_DIR
```

The default directory is:

```text
storage/images/
```

When an item is registered, the uploaded image is copied into this directory while the database stores only the image path reference.

---

## 6. Error Handling and Robustness

The application uses explicit error handling instead of silently ignoring failures.

Implemented robustness features include:

```text
no bare except blocks
no except Exception: pass
clean validation errors
clean API responses
AI retries with exponential backoff
AI timeout handling
bounded concurrency
PostgreSQL fallback handling
configurable logging
```

AI provider failures are retried with exponential backoff before being converted into application-level exceptions. Validation errors are converted into structured HTTP responses such as `422`, `404`, or `503`.

If PostgreSQL is unavailable during development, the application can fall back to JSON storage instead of crashing immediately.

---

## 7. Logging

The project uses Python’s standard `logging` module.

Logging is configurable through:

```text
LOG_LEVEL
```

The system logs important runtime events such as:

```text
database backend selection
PostgreSQL connection status
AI call timing
retry attempts
registered item IDs
batch task failures
```

CLI output uses `print` only for user-facing command results, while runtime diagnostics use structured logging.

---

## 8. Testing Architecture

Tests are implemented with `pytest` and are located in:

```text
tests/
```

The test suite covers:

```text
provided AI smoke tests
AI service behavior
matching logic
storage repositories
API end-to-end flows
concurrency behavior
error paths
```

All tests run offline using fake VLM and embedding providers. The tests do not require real API keys or network access.

Current results:

```text
91 passed
coverage: 76%
```

Coverage is measured using `pytest-cov`.

---

## 9. Deployment Architecture

The application is containerized using Docker.

The Docker image installs all pinned dependencies from `requirements.txt`, exposes port `8000`, and runs the FastAPI application through Uvicorn.

Environment variables are injected at runtime through `.env` files. This allows different configurations for local execution and Docker execution while keeping secrets outside the image itself.

---

## 10. Summary

The architecture is organized into clearly separated layers:

```text
API / CLI
Core matching service
AI service wrapper
Storage repository
Provided AI package
PostgreSQL
Filesystem image storage
```

This layered design makes the system easier to test, debug, extend, and deploy. The provided AI module remains isolated as an external dependency, while the rest of the project adds the software engineering infrastructure required for a reliable backend application.