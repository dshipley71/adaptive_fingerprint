---
name: project-scaffold
description: Scaffold, configure, and set up the Adaptive Fingerprint project from scratch. Use when initializing the project, generating configuration files, setting up the Python virtual environment, starting Redis, or running development utilities (lint, test, clean). Provides the full directory structure, pyproject.toml, requirements.txt, config.yaml generation, Docker Redis management, and development workflow commands.
---

# Project Scaffold

Set up the complete Adaptive Fingerprint project environment.

## Quick Start

```bash
chmod +x skills.sh
./skills.sh doctor       # Check prerequisites
./skills.sh scaffold     # Create project structure
./skills.sh config       # Generate config files
./skills.sh deps         # Install dependencies
./skills.sh redis        # Start Redis
source .venv/bin/activate
fingerprint analyze --url https://example.com
```

## Directory Structure

Run `./skills.sh scaffold` to generate:

```
adaptive-fingerprint/
├── fingerprint/
│   ├── __init__.py, __main__.py, config.py, models.py, exceptions.py
│   ├── core/          analyzer.py, fetcher.py, verbose.py
│   ├── adaptive/      structure_analyzer.py, change_detector.py, strategy_learner.py
│   ├── ml/            embeddings.py, ollama_client.py, classifier.py
│   ├── storage/       structure_store.py, embedding_store.py, review_store.py, cache.py
│   ├── compliance/    robots_parser.py, rate_limiter.py, bot_detector.py
│   ├── legal/         cfaa_checker.py, tos_checker.py, gdpr_handler.py, ccpa_handler.py
│   ├── extraction/    extractor.py, file_writer.py, formats.py
│   ├── alerting/      change_monitor.py, review_queue.py, notifiers.py
│   └── utils/         url_utils.py, html_utils.py
├── examples/          basic_fingerprint.py, adaptive_mode.py, ml_fingerprint.py
├── tests/             unit/, integration/
├── pyproject.toml
├── requirements.txt
├── config.yaml
└── .env
```

## Commands Reference

| Command | Purpose |
|---------|---------|
| `./skills.sh scaffold` | Create all directories and source file stubs |
| `./skills.sh deps` | Create `.venv`, install package in editable mode with dev deps |
| `./skills.sh config` | Generate `config.example.yaml`, `.env.example`, copy to `config.yaml`/`.env` |
| `./skills.sh redis` | Start Redis 7 Alpine via Docker on port 6379 |
| `./skills.sh redis-stop` | Stop the Redis container |
| `./skills.sh lint` | Run `ruff` linter on `fingerprint/` |
| `./skills.sh test` | Run `pytest` test suite |
| `./skills.sh clean` | Remove `__pycache__`, `.pytest_cache`, egg-info, `extracted/` |
| `./skills.sh clean --all` | Also remove the virtual environment |
| `./skills.sh info` | Print full architecture summary |
| `./skills.sh doctor` | Check prerequisites (Python 3.11+, Docker, Redis, Git) |

## Prerequisites

- **Required:** Python 3.11+, pip, Git
- **Optional:** Docker (for managed Redis container)

## Dependencies

See [requirements reference](references/requirements.md) for the complete dependency list.

### Core

```
httpx>=0.27.0, beautifulsoup4>=4.12.0, lxml>=5.0.0, redis>=5.0.0,
pydantic>=2.0.0, pydantic-settings>=2.0.0, pyyaml>=6.0.0
```

### ML

```
sentence-transformers>=2.3.0, numpy>=1.24.0, scikit-learn>=1.3.0
```

### CLI & Async

```
click>=8.1.0, rich>=13.0.0, anyio>=4.0.0
```

### Dev

```
pytest>=7.0.0, pytest-asyncio>=0.21.0, pytest-cov>=4.0.0, ruff>=0.1.0, mypy>=1.0.0
```

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OLLAMA_CLOUD_API_KEY` | For ML mode | - | Ollama Cloud API authentication |
| `FINGERPRINT_MODE` | No | `adaptive` | Override fingerprinting mode |
| `FINGERPRINT_VERBOSE` | No | `2` | Verbose logging level (0-3) |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection string |

## Configuration

Run `./skills.sh config` to generate config files. Edit `config.yaml` for full control over all modules. See the [config reference](references/config.md) for all available options.

## CLI Interface

```bash
fingerprint analyze   --url <URL>                     # Analyze structure
fingerprint compare   --url <URL> --mode adaptive     # Compare with stored
fingerprint describe  --url <URL>                     # Generate LLM description
fingerprint extract   --url <URL> --format json       # Extract content
fingerprint review    list --limit 50                 # View review queue
fingerprint review    approve <id> --notes "OK"       # Approve change
fingerprint review    reject <id> --notes "False pos"  # Reject change
fingerprint review    stats                           # Queue statistics
fingerprint -vvv analyze --url <URL>                  # Maximum verbosity
```

---

# Configuration Reference

Complete `config.yaml` options for the Adaptive Fingerprint system.

## Fingerprinting

```yaml
fingerprinting:
  mode: adaptive                          # "rules", "ml", or "adaptive"
  adaptive:
    class_change_threshold: 0.15          # Escalate to ML if >15% classes changed
    rules_uncertainty_threshold: 0.80     # Escalate to ML if rules similarity < 0.80
    cache_ml_results: true                # Cache ML results to avoid recomputation
  thresholds:
    cosmetic: 0.95                        # > 0.95 = cosmetic change
    minor: 0.85                           # 0.85-0.95 = minor change
    moderate: 0.70                        # 0.70-0.85 = moderate change
    breaking: 0.70                        # < 0.70 = breaking change
```

## Ollama Cloud

```yaml
ollama_cloud:
  enabled: true                           # Enable LLM descriptions
  model: "gemma3:12b"                     # Model to use
  timeout: 30                             # Request timeout (seconds)
  max_retries: 3                          # Max retry attempts
  temperature: 0.3                        # LLM temperature
  max_tokens: 500                         # Max response tokens
```

## Embeddings

```yaml
embeddings:
  model: "all-MiniLM-L6-v2"              # Sentence transformer model
  cache_embeddings: true                  # Cache generated embeddings
```

## Redis

```yaml
redis:
  url: "redis://localhost:6379/0"         # Connection URL
  key_prefix: "fingerprint"              # Key namespace prefix
  ttl_seconds: 604800                     # Key expiration (7 days)
  max_versions: 10                        # Max structure versions to keep
```

## HTTP

```yaml
http:
  user_agent: "AdaptiveFingerprint/1.0 (+https://example.com/bot-info)"
  timeout: 30
  max_retries: 3
```

## Compliance

```yaml
compliance:
  robots_txt:
    enabled: true
    cache_ttl: 3600                       # Cache robots.txt for 1 hour
    respect_crawl_delay: true
    default_crawl_delay: 1.0
  rate_limiting:
    enabled: true
    default_delay: 1.0
    min_delay: 0.5
    max_delay: 30.0
    backoff_multiplier: 2.0
    adapt_to_response_time: true
  anti_bot:
    enabled: true
    respect_retry_after: true
    stop_on_captcha: true
    stop_on_block_page: true
```

## Legal

```yaml
legal:
  cfaa:
    enabled: true
    require_public_access: true
    block_authenticated_areas: true
    block_api_endpoints: true
  tos:
    enabled: true
    check_meta_tags: true
    respect_noindex: true
    respect_nofollow: true
  gdpr:
    enabled: true
    pii_detection: true
    pii_handling: "redact"                # "redact", "pseudonymize", or "skip"
    log_pii_access: true
  ccpa:
    enabled: true
    respect_opt_out: true
    respect_gpc: true
```

## Extraction

```yaml
extraction:
  enabled: true
  output_dir: "./extracted"
  formats: ["json", "csv"]
  include_metadata: true
  include_html: false
  max_content_length: 1000000
```

## Alerting

```yaml
alerting:
  enabled: true
  alert_on_breaking: true
  alert_on_moderate: false
  alert_threshold: 0.70
  review_queue:
    enabled: true
    auto_approve_cosmetic: true
    auto_approve_minor: false
    require_review_breaking: true
    max_queue_size: 1000
  notifications:
    log: true
    webhook:
      enabled: false
      url: ""
    email:
      enabled: false
      smtp_host: ""
      smtp_port: 587
      recipients: []
```

## Verbose

```yaml
verbose:
  enabled: true
  level: 2                                # 0=errors, 1=warnings, 2=info, 3=debug
  format: "structured"                    # "structured" or "plain"
  include_timestamp: true
```

---

# Requirements Reference

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `httpx` | >=0.27.0 | Async HTTP client |
| `beautifulsoup4` | >=4.12.0 | HTML parsing |
| `lxml` | >=5.0.0 | Fast XML/HTML parser backend |
| `redis` | >=5.0.0 | Redis client (async support) |
| `pydantic` | >=2.0.0 | Data validation |
| `pydantic-settings` | >=2.0.0 | Settings management |
| `pyyaml` | >=6.0.0 | YAML config parsing |

## ML Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | >=2.3.0 | Embedding generation |
| `numpy` | >=1.24.0 | Vector operations |
| `scikit-learn` | >=1.3.0 | ML utilities |

## Optional ML Backends

| Package | Version | Purpose |
|---------|---------|---------|
| `xgboost` | >=2.0.0 | Gradient boosting |
| `lightgbm` | >=4.0.0 | Light gradient boosting |

## Async & CLI

| Package | Version | Purpose |
|---------|---------|---------|
| `anyio` | >=4.0.0 | Async compatibility |
| `click` | >=8.1.0 | CLI framework |
| `rich` | >=13.0.0 | Terminal formatting |

## Dev Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=7.0.0 | Test framework |
| `pytest-asyncio` | >=0.21.0 | Async test support |
| `pytest-cov` | >=4.0.0 | Coverage reporting |
| `ruff` | >=0.1.0 | Linter and formatter |
| `mypy` | >=1.0.0 | Type checking |

## Python Version

Requires Python **3.11+** (uses `X | Y` union syntax and `match` statements).
