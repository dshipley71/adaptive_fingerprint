---
name: core-orchestration
description: Implement the core orchestration layer for the Adaptive Fingerprint system. Use when building or modifying the main StructureAnalyzer, ComplianceFetcher, or VerboseLogger classes in fingerprint/core/. Covers the central coordinator that routes between rules-based, ML-based, and adaptive fingerprinting modes, the compliance-aware HTTP fetcher pipeline, and the structured verbose logging system used by every module.
---

# Core Orchestration

Implement `fingerprint/core/analyzer.py`, `fingerprint/core/fetcher.py`, and `fingerprint/core/verbose.py`.

## Files

| File | Class | Purpose |
|------|-------|---------|
| `analyzer.py` | `StructureAnalyzer` | Main orchestrator coordinating all fingerprinting operations |
| `fetcher.py` | `HTTPFetcher` | Low-level HTTP client (internal only, never exposed) |
| `fetcher.py` | `ComplianceFetcher` | Wraps HTTPFetcher with the full compliance pipeline |
| `verbose.py` | `VerboseLogger` | Structured logging with module prefixes |
| `verbose.py` | `get_logger()` / `set_logger()` | Module-level logger access |

## StructureAnalyzer

Central coordinator. Initialize with a `Config` object, which wires up all sub-components:

```python
class StructureAnalyzer:
    def __init__(self, config: Config):
        self.fetcher = HTTPFetcher(config.http)
        self.dom_analyzer = DOMStructureAnalyzer()
        self.change_detector = ChangeDetector(config.thresholds)
        self.embedding_generator = EmbeddingGenerator(config.embeddings)
        self.ollama_client = OllamaCloudClient(config.ollama_cloud)
        self.structure_store = StructureStore(config.redis)
```

### Key Methods

- `analyze_url(url) -> PageStructure` - Fetch and produce a DOM fingerprint.
- `compare_with_stored(url) -> ChangeAnalysis` - Compare current vs. stored structure using the configured mode.
- `generate_description(structure) -> str` - Call Ollama Cloud for a human-readable description.

### Mode Dispatch

`compare_with_stored` dispatches based on `config.mode`:

| Mode | Method | Latency |
|------|--------|---------|
| `rules` | `_compare_rules()` | ~15 ms |
| `ml` | `_compare_ml()` | ~200 ms |
| `adaptive` | `_compare_adaptive()` | 15-200 ms |

### Adaptive Mode Logic

Implement `_compare_adaptive` with this sequence:

1. Run rules-based comparison first.
2. Check escalation triggers against the rules result.
3. If any trigger fires, escalate to ML and return the ML result instead.
4. If no trigger fires, return the rules result.

Escalation triggers:

| Trigger | Condition |
|---------|-----------|
| `CLASS_VOLATILITY` | >15% of CSS classes changed between stored and current |
| `RULES_UNCERTAINTY` | Rules similarity score < 0.80 |
| `KNOWN_VOLATILE` | Domain flagged volatile in Redis (`{prefix}:volatile:{domain}`) |
| `RENAME_PATTERN` | Detected class rename patterns (e.g., `btn-primary` -> `btn-primary-v2`) |

When escalating, record `EscalationTrigger` objects on the `ChangeAnalysis.escalation_triggers` list and set `escalated=True`.

### First-Visit Behavior

When `compare_with_stored` finds no stored structure, save the current structure and return a `ChangeAnalysis` with `similarity=1.0`, `classification=COSMETIC`, `breaking=False`.

### Breaking Change Handling

When a breaking change is detected (`classification == BREAKING`), automatically save the new structure as the latest version.

## ComplianceFetcher Pipeline

Every HTTP fetch passes through this mandatory pipeline in order:

```
1. CFAA Check        -> CFAAChecker.is_authorized(url)
2. ToS Check         -> ToSChecker.check(url)
3. robots.txt Check  -> RobotsChecker.is_allowed(url)
4. Rate Limiter      -> RateLimiter.acquire(domain)
5. HTTP Fetch        -> HTTPFetcher.fetch(url)
6. Anti-Bot Check    -> BotDetector.check(response)
7. GDPR/CCPA Check   -> GDPRHandler.process(response), CCPAHandler.process(response)
```

If any check fails, raise the corresponding exception and do not proceed. See [compliance pipeline](references/compliance-pipeline.md) for full details.

## VerboseLogger

All modules log through this centralized system with a consistent format:

```
[2024-01-15T10:30:00Z] [MODULE:OPERATION] Message
  - detail_key: value
```

Levels: `0`=errors only, `1`=warnings, `2`=info, `3`=debug. Controlled by `FINGERPRINT_VERBOSE` env var or `config.verbose.level`.

Module prefixes used across the system:

```
ANALYZER  STRUCTURE  CHANGE   ADAPTIVE  ML       OLLAMA
STORE     ROBOTS     RATELIMIT ANTIBOT  CFAA     TOS
GDPR      CCPA       EXTRACT  FILEWRITER ALERT   REVIEW
NOTIFY    CLASSIFY   CACHE    FETCH     EMBED_STORE
```

## Data Flow Summary

```
URL -> ComplianceFetcher -> HTML
HTML -> DOMStructureAnalyzer -> PageStructure
PageStructure -> StructureStore (save/load)
PageStructure + StoredPageStructure -> ChangeDetector -> ChangeAnalysis
PageStructure -> EmbeddingGenerator -> StructureEmbedding (ML mode)
PageStructure -> OllamaCloudClient -> description string (optional)
```
