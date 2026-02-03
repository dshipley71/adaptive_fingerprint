---
name: ml-integration
description: Implement the ML embeddings and Ollama Cloud LLM integration for the Adaptive Fingerprint system. Use when building or modifying fingerprint/ml/ modules including EmbeddingGenerator, OllamaCloudClient, and PageClassifier. Covers sentence-transformer embedding generation (all-MiniLM-L6-v2), cosine similarity calculation, Ollama Cloud API chat endpoint integration for human-readable structure descriptions, and page type classification using reference embeddings.
---

# ML Integration

Implement `fingerprint/ml/embeddings.py`, `fingerprint/ml/ollama_client.py`, and `fingerprint/ml/classifier.py`.

## EmbeddingGenerator

Generate semantic vector embeddings from `PageStructure` objects using sentence-transformers.

```python
class EmbeddingGenerator:
    def __init__(self, config: EmbeddingsConfig): ...

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""

    async def generate(self, structure: PageStructure) -> StructureEmbedding: ...
    def cosine_similarity(self, vector1: list[float], vector2: list[float]) -> float: ...
```

### Configuration

- Default model: `all-MiniLM-L6-v2` (384 dimensions)
- Normalize embeddings: `True`
- Lazy loading: model loads on first call to `.model`

### Structure Description Format

Convert `PageStructure` to text for embedding with `_create_description`:

```
Page type: {type}. Framework: {framework}. Landmarks: {landmarks}.
Main tags: {tag}({count}), ... (top 10). Key classes: {class}, ... (top 15).
Content regions: {regions}.
```

### Cosine Similarity

Compute manually with numpy:

```python
dot_product = np.dot(v1, v2)
similarity = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

Return `0.0` if either vector has zero norm.

## OllamaCloudClient

HTTP client for the Ollama Cloud chat API.

```python
class OllamaCloudClient:
    API_URL = "https://ollama.com/api/chat"

    async def describe_structure(self, structure: PageStructure) -> str: ...
    async def analyze_change(self, old: PageStructure, new: PageStructure) -> str: ...
```

### API Details

| Field | Value |
|-------|-------|
| Endpoint | `POST https://ollama.com/api/chat` |
| Auth | `Authorization: Bearer {OLLAMA_CLOUD_API_KEY}` |
| Default model | `gemma3:12b` |
| Stream | `false` |

### Request Payload

```json
{
    "model": "gemma3:12b",
    "messages": [{"role": "user", "content": "<prompt>"}],
    "stream": false,
    "options": {"num_predict": 500, "temperature": 0.3}
}
```

### Response Extraction

Extract `response["message"]["content"]` as the description string.

### Error Handling

| Status | Exception | Retry? |
|--------|-----------|--------|
| 401 | `OllamaAuthError` | No |
| 429 | `OllamaRateLimitError` | No |
| 500+ | `OllamaCloudError` | Yes (up to `max_retries`) |
| Timeout | `OllamaTimeoutError` | Yes |

Retry with exponential backoff for retriable errors only.

### Prompt Templates

**Structure description prompt** - ask for a 2-3 sentence description covering:
1. What type of content the page likely contains
2. How the page is organized (landmarks, regions)
3. Notable structural patterns

**Change analysis prompt** - provide added/removed classes, landmark changes, framework changes, and ask for:
1. What likely changed (redesign, CSS refactor, framework update)
2. Impact on content extraction (breaking or compatible)
3. Recommended action (adapt selectors, learn new strategy, flag for review)

## PageClassifier

Classify pages into types using embedding similarity against reference descriptions.

```python
class PageClassifier:
    REFERENCE_DESCRIPTIONS = {
        "article": "Page type: article. Landmarks: header, nav, main, footer, article. ...",
        "listing": "Page type: listing. ...",
        "product": "Page type: product. ...",
        "home": "Page type: home. ...",
    }

    async def classify(self, structure: PageStructure) -> str: ...
    async def classify_with_confidence(self, structure: PageStructure) -> tuple[str, float]: ...
```

### Classification Flow

1. Generate embedding for the input structure.
2. Lazy-load reference embeddings (encode each `REFERENCE_DESCRIPTIONS` value once).
3. Compute cosine similarity against each reference.
4. Return the page type with the highest similarity.

## Verbose Logging

| Operation | Description |
|-----------|-------------|
| `ML:INIT` | Generator initialized |
| `ML:LOAD` | Loading sentence-transformer model |
| `ML:EMBED` | Generating embedding for a structure |
| `ML:VECTOR` | Embedding generated (dimensions, norm) |
| `ML:SIMILARITY` | Cosine similarity computed |
| `OLLAMA:REQUEST` | Sending API request |
| `OLLAMA:RESPONSE` | Response received |
| `OLLAMA:TIMEOUT` | Request timed out |
| `CLASSIFY:START` | Classification started |
| `CLASSIFY:COMPARE` | Similarity against each reference |
| `CLASSIFY:RESULT` | Final classification with score |
