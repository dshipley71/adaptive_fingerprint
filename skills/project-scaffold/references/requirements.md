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
