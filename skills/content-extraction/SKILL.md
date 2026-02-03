---
name: content-extraction
description: Implement the content extraction and file output system for the Adaptive Fingerprint system. Use when building or modifying fingerprint/extraction/ modules including ContentExtractor, FileWriter, and output format handlers (JSON, CSV, Markdown). Covers CSS selector-based content extraction using learned strategies, multi-format file persistence with metadata inclusion, and confidence-tracked extraction results.
---

# Content Extraction

Implement `fingerprint/extraction/extractor.py`, `fingerprint/extraction/file_writer.py`, and `fingerprint/extraction/formats.py`.

## ContentExtractor

Extract content from HTML using learned `ExtractionStrategy` objects.

```python
class ContentExtractor:
    def __init__(self, config: Config): ...
    async def extract(self, url: str) -> ExtractionResult: ...
    async def extract_with_structure(
        self, html: str, structure: PageStructure, strategy: ExtractionStrategy,
    ) -> ExtractionResult: ...
```

### Extraction Pipeline

1. Look up the `ExtractionStrategy` for the domain/page_type (from store or from `StrategyLearner`).
2. Parse HTML with `BeautifulSoup(html, "lxml")`.
3. For each field in the strategy (`title`, `content`, `metadata`):
   a. Try the `primary` CSS selector.
   b. If no match, try each `fallback` selector in order.
   c. Apply `extraction_method`: `"text"` (`.get_text()`), `"html"` (`.decode_contents()`), or `"attribute"` (`.get(attribute_name)`).
   d. Apply `post_processors` in order (e.g., `strip`, `normalize_whitespace`, `remove_html`).
   e. Record which selector succeeded and the confidence score.
4. Build and return `ExtractionResult`.

### ExtractionResult Fields

```python
@dataclass
class ExtractionResult:
    success: bool
    content: ExtractedContent | None = None
    error: str = ""
    fields_extracted: int = 0
    content_length: int = 0
    duration_ms: float = 0.0
```

### ExtractedContent Fields

```python
@dataclass
class ExtractedContent:
    url: str
    domain: str
    page_type: str
    title: str = ""
    content: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    html: str = ""                        # Only if include_html=True
    extracted_at: datetime = ...
    strategy_version: int = 1
    extraction_confidence: float = 0.0
    output_file: str | None = None
```

## FileWriter

Persist extracted content to disk.

```python
class FileWriter:
    def __init__(self, config: ExtractionConfig): ...
    async def save(self, content: ExtractedContent, format: str) -> str: ...
    async def save_batch(self, contents: list[ExtractedContent], format: str) -> list[str]: ...
    def register_formatter(self, name: str, formatter: OutputFormatter) -> None: ...
```

### Output Path Convention

```
{output_dir}/{domain}/{page_type}/{date}_{hash}.{format}
```

Example: `extracted/example.com/article/2024-01-15_a3f2b1c4.json`

- `date` = `YYYY-MM-DD`
- `hash` = first 8 chars of SHA-256 of the URL

### Metadata Inclusion

When `include_metadata=True`, append extraction metadata to output:

```json
{
  "_metadata": {
    "extracted_at": "2024-01-15T10:30:00Z",
    "strategy_version": 1,
    "extraction_confidence": 0.92,
    "source_url": "https://example.com/article",
    "fingerprint_mode": "adaptive"
  }
}
```

## Output Formats

```python
class OutputFormatter(ABC):
    @abstractmethod
    def format(self, content: ExtractedContent) -> str: ...
    @abstractmethod
    def format_batch(self, contents: list[ExtractedContent]) -> str: ...
    @abstractmethod
    def file_extension(self) -> str: ...
```

### JSONFormatter

Standard JSON output with indentation. Batch mode produces a JSON array.

### CSVFormatter

Flat CSV with columns: `url`, `domain`, `page_type`, `title`, `content`, plus each metadata key. Batch mode produces a single CSV with header row.

### MarkdownFormatter

Human-readable Markdown output:

```markdown
# {title}

**Source:** {url}
**Type:** {page_type}
**Extracted:** {extracted_at}

---

{content}

---

## Metadata

| Field | Value |
|-------|-------|
| author | ... |
| date | ... |
```

## Configuration

```yaml
extraction:
  enabled: true
  output_dir: "./extracted"
  formats: ["json", "csv"]
  include_metadata: true
  include_html: false
  max_content_length: 1000000
```

## Verbose Logging

| Operation | Description |
|-----------|-------------|
| `EXTRACT:START` | Starting extraction |
| `EXTRACT:STRATEGY` | Using strategy (domain, page_type, version) |
| `EXTRACT:SELECTOR` | Trying CSS selector |
| `EXTRACT:MATCH` | Selector matched |
| `EXTRACT:FALLBACK` | Primary failed, trying fallback |
| `EXTRACT:FIELD` | Field extracted (name, length, confidence) |
| `EXTRACT:RESULT` | Extraction complete (fields, duration) |
| `FILEWRITER:SAVE` | Saving to file (path, format) |
| `FILEWRITER:BATCH` | Saving batch (count, format) |
