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
