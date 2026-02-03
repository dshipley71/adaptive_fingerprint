#!/usr/bin/env bash
# =============================================================================
# skills.sh - Adaptive Structure Fingerprinting System
# =============================================================================
#
# Project setup, scaffolding, and development utilities derived from
# the AGENTS.md specification files.
#
# Usage:
#   chmod +x skills.sh
#   ./skills.sh <command>
#
# Commands:
#   scaffold      - Create full project directory structure and source files
#   deps          - Install Python dependencies into a virtual environment
#   config        - Generate config.example.yaml and .env.example
#   redis         - Start Redis via Docker for structure storage
#   redis-stop    - Stop the Redis container
#   lint          - Run linting on the fingerprint package
#   test          - Run the test suite
#   clean         - Remove generated artifacts (venv, __pycache__, extracted/)
#   info          - Display project architecture summary
#   doctor        - Verify all prerequisites are installed
#   help          - Show this help message
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
REDIS_CONTAINER_NAME="adaptive-fingerprint-redis"
REDIS_PORT=6379

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BOLD}==> $*${NC}"; }

# ===========================================================================
# scaffold - Create full project directory structure
# ===========================================================================
cmd_scaffold() {
    log_step "Scaffolding Adaptive Fingerprint project structure"

    # -----------------------------------------------------------------------
    # Package directories
    # -----------------------------------------------------------------------
    local dirs=(
        "fingerprint"
        "fingerprint/core"
        "fingerprint/adaptive"
        "fingerprint/ml"
        "fingerprint/storage"
        "fingerprint/compliance"
        "fingerprint/legal"
        "fingerprint/extraction"
        "fingerprint/alerting"
        "fingerprint/utils"
        "examples"
        "tests"
        "tests/unit"
        "tests/integration"
    )

    for d in "${dirs[@]}"; do
        mkdir -p "${PROJECT_ROOT}/${d}"
        log_info "Created directory: ${d}/"
    done

    # -----------------------------------------------------------------------
    # __init__.py stubs
    # -----------------------------------------------------------------------
    _write_if_missing "fingerprint/__init__.py" '"""Adaptive Structure Fingerprinting System."""

__version__ = "1.0.0"
'

    _write_if_missing "fingerprint/core/__init__.py" '"""
Core module - Main orchestration for fingerprinting system.
"""

from fingerprint.core.analyzer import StructureAnalyzer
from fingerprint.core.fetcher import HTTPFetcher
from fingerprint.core.verbose import VerboseLogger, get_logger, set_logger

__all__ = [
    "StructureAnalyzer",
    "HTTPFetcher",
    "VerboseLogger",
    "get_logger",
    "set_logger",
]
'

    _write_if_missing "fingerprint/adaptive/__init__.py" '"""
Adaptive module - Rules-based fingerprinting and change detection.
"""

from fingerprint.adaptive.structure_analyzer import DOMStructureAnalyzer
from fingerprint.adaptive.change_detector import ChangeDetector
from fingerprint.adaptive.strategy_learner import StrategyLearner

__all__ = [
    "DOMStructureAnalyzer",
    "ChangeDetector",
    "StrategyLearner",
]
'

    _write_if_missing "fingerprint/ml/__init__.py" '"""
ML module - Embeddings and Ollama Cloud integration.
"""

from fingerprint.ml.embeddings import EmbeddingGenerator
from fingerprint.ml.ollama_client import OllamaCloudClient
from fingerprint.ml.classifier import PageClassifier

__all__ = [
    "EmbeddingGenerator",
    "OllamaCloudClient",
    "PageClassifier",
]
'

    _write_if_missing "fingerprint/storage/__init__.py" '"""
Storage module - Redis persistence layer.
"""

from fingerprint.storage.structure_store import StructureStore
from fingerprint.storage.embedding_store import EmbeddingStore
from fingerprint.storage.review_store import ReviewStore
from fingerprint.storage.cache import Cache

__all__ = [
    "StructureStore",
    "EmbeddingStore",
    "ReviewStore",
    "Cache",
]
'

    _write_if_missing "fingerprint/compliance/__init__.py" '"""
Compliance module - Ethical web access.

Provides robots.txt parsing, rate limiting, and anti-bot detection.
"""

from fingerprint.compliance.robots_parser import RobotsParser, RobotsChecker
from fingerprint.compliance.rate_limiter import RateLimiter, DomainState
from fingerprint.compliance.bot_detector import BotDetector, BotCheckResult

__all__ = [
    "RobotsParser",
    "RobotsChecker",
    "RateLimiter",
    "DomainState",
    "BotDetector",
    "BotCheckResult",
]
'

    _write_if_missing "fingerprint/legal/__init__.py" '"""
Legal compliance module.

Provides CFAA, ToS, GDPR, and CCPA compliance checking.
"""

from fingerprint.legal.cfaa_checker import CFAAChecker, AuthorizationResult
from fingerprint.legal.tos_checker import ToSChecker, ToSResult
from fingerprint.legal.gdpr_handler import GDPRHandler, PIIDetectionResult
from fingerprint.legal.ccpa_handler import CCPAHandler

__all__ = [
    "CFAAChecker",
    "AuthorizationResult",
    "ToSChecker",
    "ToSResult",
    "GDPRHandler",
    "PIIDetectionResult",
    "CCPAHandler",
]
'

    _write_if_missing "fingerprint/extraction/__init__.py" '"""
Extraction module - Content extraction and file saving.
"""

from fingerprint.extraction.extractor import ContentExtractor
from fingerprint.extraction.file_writer import FileWriter
from fingerprint.extraction.formats import JSONFormatter, CSVFormatter, OutputFormatter

__all__ = [
    "ContentExtractor",
    "FileWriter",
    "JSONFormatter",
    "CSVFormatter",
    "OutputFormatter",
]
'

    _write_if_missing "fingerprint/alerting/__init__.py" '"""
Alerting module - Change alerts and manual review queue.
"""
'

    _write_if_missing "fingerprint/utils/__init__.py" '"""
Utilities module.
"""
'

    _write_if_missing "tests/__init__.py" ""
    _write_if_missing "tests/unit/__init__.py" ""
    _write_if_missing "tests/integration/__init__.py" ""

    # -----------------------------------------------------------------------
    # Source file stubs (empty modules with docstrings)
    # Only created if they do not already exist.
    # -----------------------------------------------------------------------

    # -- Core --
    _write_if_missing "fingerprint/models.py" '"""
Core data models for the fingerprinting system.

All models use dataclasses for clean, type-safe data structures.
Enums: FingerprintMode, ChangeClassification, ChangeType, ReviewStatus, AlertSeverity
Dataclasses: PageStructure, StructureEmbedding, ChangeAnalysis, ExtractionStrategy,
             ExtractedContent, ExtractionResult, ReviewItem, ChangeAlert, and more.
"""
'

    _write_if_missing "fingerprint/config.py" '"""
Configuration management using Pydantic settings.

Loads configuration from:
1. Environment variables (FINGERPRINT_* prefix)
2. YAML configuration file
3. Default values

Key classes: FingerprintSettings, Config, load_config()
"""
'

    _write_if_missing "fingerprint/exceptions.py" '"""
Custom exception hierarchy for the fingerprinting system.

All exceptions inherit from FingerprintError for easy catching.
Hierarchy:
  FingerprintError
    AnalysisError -> InvalidHTMLError, EmptyContentError
    ChangeDetectionError -> IncompatibleStructuresError
    MLError -> EmbeddingError, ModelLoadError
    OllamaCloudError -> OllamaAuthError, OllamaTimeoutError, OllamaRateLimitError
    StorageError -> RedisConnectionError, SerializationError
    FetchError -> HTTPTimeoutError, HTTPStatusError
    ComplianceError -> RobotsBlockedError, RateLimitExceededError, CrawlDelayError,
                       BotDetectedError -> CaptchaEncounteredError
    LegalComplianceError -> CFAAViolationError -> UnauthorizedAccessError,
                            ToSViolationError, GDPRViolationError, CCPAViolationError
"""
'

    _write_if_missing "fingerprint/__main__.py" '"""
CLI entry point for the fingerprinting system.

Usage:
    fingerprint analyze --url https://example.com
    fingerprint compare --url https://example.com --mode adaptive
    fingerprint describe --url https://example.com
    fingerprint extract --url https://example.com --output ./extracted --format json
    fingerprint review list --limit 50
    fingerprint review approve <item-id> --notes "Verified"
"""
'

    # -- Core module --
    _write_if_missing "fingerprint/core/analyzer.py" '"""
Main analyzer orchestrator.

Coordinates fingerprinting operations across all modes:
- Rules-based: Uses DOMStructureAnalyzer from adaptive module
- ML-based: Uses embeddings and Ollama Cloud from ml module
- Adaptive: Intelligently selects mode based on escalation triggers
  (CLASS_VOLATILITY, RULES_UNCERTAINTY, KNOWN_VOLATILE, RENAME_PATTERN)

Key class: StructureAnalyzer
"""
'

    _write_if_missing "fingerprint/core/fetcher.py" '"""
HTTP fetcher with full compliance pipeline.

All fetches pass through the ethical compliance pipeline:
1. CFAA authorization check
2. ToS check
3. robots.txt check (RFC 9309)
4. Rate limiting with Crawl-delay
5. HTTP fetch
6. Anti-bot detection
7. GDPR/CCPA compliance

Key classes: ComplianceFetcher, HTTPFetcher (internal only), FetchResult
"""
'

    _write_if_missing "fingerprint/core/verbose.py" '"""
Verbose logging system with structured output.

All modules use this for consistent logging format:
[TIMESTAMP] [MODULE:OPERATION] Message
  - detail_1
  - detail_2

Levels: 0=errors, 1=warnings, 2=info, 3=debug

Key class: VerboseLogger
Functions: get_logger(), set_logger()
"""
'

    # -- Adaptive module --
    _write_if_missing "fingerprint/adaptive/structure_analyzer.py" '"""
Rules-based DOM structure analysis.

Analyzes HTML to produce a PageStructure fingerprint:
- Tag hierarchy and depth distribution
- CSS class map and ID attributes
- Semantic landmarks (header, nav, main, footer, etc.)
- Content region detection
- Script signature analysis and framework detection

Key class: DOMStructureAnalyzer
"""
'

    _write_if_missing "fingerprint/adaptive/change_detector.py" '"""
Change detection and classification.

Compares two PageStructure objects and produces a ChangeAnalysis:
- Tag similarity (Jaccard coefficient)
- Class similarity
- Landmark comparison
- Change classification: cosmetic (>0.95), minor (0.85-0.95),
  moderate (0.70-0.85), breaking (<0.70)

Key class: ChangeDetector
"""
'

    _write_if_missing "fingerprint/adaptive/strategy_learner.py" '"""
CSS selector inference for content extraction.

Learns extraction strategies from page structures by:
- Identifying content regions via landmark analysis
- Inferring CSS selectors with fallbacks
- Tracking confidence scores

Key class: StrategyLearner
"""
'

    # -- ML module --
    _write_if_missing "fingerprint/ml/embeddings.py" '"""
Embedding generation for semantic fingerprinting.

Uses sentence-transformers to generate embeddings from structure descriptions.
Default model: all-MiniLM-L6-v2 (384 dimensions)

Key class: EmbeddingGenerator
Methods: generate(), cosine_similarity()
"""
'

    _write_if_missing "fingerprint/ml/ollama_client.py" '"""
Ollama Cloud API client for LLM-powered descriptions.

Endpoint: POST https://ollama.com/api/chat
Authentication: Bearer token (OLLAMA_CLOUD_API_KEY)
Default model: gemma3:12b

Key class: OllamaCloudClient
Methods: describe_structure(), analyze_change()
"""
'

    _write_if_missing "fingerprint/ml/classifier.py" '"""
Page type classification using embeddings.

Uses cosine similarity with reference embeddings to classify page types:
article, listing, product, home

Key class: PageClassifier
Methods: classify(), classify_with_confidence()
"""
'

    # -- Storage module --
    _write_if_missing "fingerprint/storage/structure_store.py" '"""
Redis storage for page structures.

Key patterns:
- {prefix}:structure:{domain}:{page_type}:{variant_id} - Current
- {prefix}:structure:{domain}:{page_type}:{variant_id}:v{n} - History
- {prefix}:volatile:{domain} - Volatile domain flag

Key class: StructureStore
Methods: save(), get(), delete(), list_versions(), is_volatile(), mark_volatile()
"""
'

    _write_if_missing "fingerprint/storage/embedding_store.py" '"""
Redis storage for structure embeddings.

Key pattern: {prefix}:embedding:{domain}:{page_type}:{variant_id}

Key class: EmbeddingStore
Methods: save(), get(), delete()
"""
'

    _write_if_missing "fingerprint/storage/review_store.py" '"""
Redis storage for review queue items.

Key patterns:
- {prefix}:review:pending              - Sorted set (ID -> timestamp)
- {prefix}:review:item:{id}            - Individual review item
- {prefix}:review:domain:{domain}      - Set of review IDs per domain
- {prefix}:review:completed:{id}       - Archived completed reviews

Key class: ReviewStore
Methods: add(), get(), get_pending(), approve(), reject(), stats()
"""
'

    _write_if_missing "fingerprint/storage/cache.py" '"""
In-memory TTL-based caching utilities.

Caches expensive operations: embedding generation, Ollama Cloud responses,
structure comparisons.

Key class: Cache[T]
Methods: set(), get(), delete(), clear(), cleanup_expired(), stats()
"""
'

    # -- Compliance module --
    _write_if_missing "fingerprint/compliance/robots_parser.py" '"""
robots.txt parser compliant with RFC 9309.

Key features:
- Full RFC 9309 compliance
- Crawl-delay support
- Sitemap discovery
- Caching with TTL
- Wildcard pattern matching (* and $)

Key classes: RobotsParser, RobotsChecker, RobotsRule, RobotsData
"""
'

    _write_if_missing "fingerprint/compliance/rate_limiter.py" '"""
Adaptive rate limiter with per-domain tracking.

Features:
- Per-domain delay tracking
- Crawl-delay respect from robots.txt
- Automatic backoff on errors/429s
- Adaptive delay based on response times
- Exponential backoff with configurable multiplier

Key classes: RateLimiter, DomainState
"""
'

    _write_if_missing "fingerprint/compliance/bot_detector.py" '"""
Anti-bot detection and respect.

Detects when the site has identified us as a bot and responds appropriately.
We RESPECT anti-bot measures rather than trying to evade them.

Detection types: CAPTCHA, block pages, rate limits, JS challenges

Key classes: BotDetector, BotCheckResult
"""
'

    # -- Legal module --
    _write_if_missing "fingerprint/legal/cfaa_checker.py" '"""
CFAA (Computer Fraud and Abuse Act) authorization checker.

Blocks: login areas, API endpoints, internal/system paths,
        auth query parameters, non-HTTP(S) schemes.

Key classes: CFAAChecker, AuthorizationResult
"""
'

    _write_if_missing "fingerprint/legal/tos_checker.py" '"""
Terms of Service compliance checker.

Respects: meta robots tags (noindex, nofollow, noarchive),
          X-Robots-Tag headers.

Key classes: ToSChecker, ToSResult
"""
'

    _write_if_missing "fingerprint/legal/gdpr_handler.py" '"""
GDPR (General Data Protection Regulation) compliance handler.

PII detection: email, phone, IP, SSN, credit card, postal code, EU phone.
Handling modes: redact, pseudonymize, skip.

Key classes: GDPRHandler, PIIDetectionResult, PIIMatch
"""
'

    _write_if_missing "fingerprint/legal/ccpa_handler.py" '"""
CCPA (California Consumer Privacy Act) compliance handler.

Respects: GPC (Global Privacy Control) signal, "Do Not Sell" opt-outs.

Key classes: CCPAHandler, CCPACheckResult
"""
'

    # -- Extraction module --
    _write_if_missing "fingerprint/extraction/extractor.py" '"""
Content extraction engine.

Extracts content from HTML using learned extraction strategies with
CSS selectors, fallbacks, and post-processors.

Key class: ContentExtractor
Methods: extract(), extract_with_structure()
"""
'

    _write_if_missing "fingerprint/extraction/file_writer.py" '"""
File writer for extracted content.

Saves extracted content to files in various formats.
Output structure: extracted/{domain}/{page_type}/{date}_{hash}.{format}

Key class: FileWriter
Methods: save(), save_batch(), register_formatter()
"""
'

    _write_if_missing "fingerprint/extraction/formats.py" '"""
Output format handlers for extracted content.

Formats: JSON, CSV, Markdown

Key classes: OutputFormatter (ABC), JSONFormatter, CSVFormatter, MarkdownFormatter
"""
'

    # -- Alerting module --
    _write_if_missing "fingerprint/alerting/change_monitor.py" '"""
Monitor for breaking changes.

Detects structure changes that exceed alert thresholds and creates alerts.

Key class: ChangeMonitor
"""
'

    _write_if_missing "fingerprint/alerting/review_queue.py" '"""
Manual review queue for structure changes.

Auto-approves cosmetic changes, requires review for breaking changes.

Key class: ReviewQueue
"""
'

    _write_if_missing "fingerprint/alerting/notifiers.py" '"""
Alert notification channels: log, webhook, email.

Key classes: LogNotifier, WebhookNotifier, EmailNotifier
"""
'

    # -- Utils module --
    _write_if_missing "fingerprint/utils/url_utils.py" '"""
URL normalization and domain extraction utilities.
"""
'

    _write_if_missing "fingerprint/utils/html_utils.py" '"""
HTML parsing utilities.
"""
'

    # -----------------------------------------------------------------------
    # pyproject.toml
    # -----------------------------------------------------------------------
    _write_if_missing "pyproject.toml" '[project]
name = "adaptive-fingerprint"
version = "1.0.0"
description = "Adaptive web structure fingerprinting with ML and Ollama Cloud integration"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}

dependencies = [
    "httpx>=0.27.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=5.0.0",
    "redis>=5.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "pyyaml>=6.0.0",
    "sentence-transformers>=2.3.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "anyio>=4.0.0",
    "click>=8.1.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
ml-backends = [
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.scripts]
fingerprint = "fingerprint.__main__:main"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.ruff]
target-version = "py311"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
'

    # -----------------------------------------------------------------------
    # requirements.txt
    # -----------------------------------------------------------------------
    _write_if_missing "requirements.txt" '# Core dependencies
httpx>=0.27.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
redis>=5.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
pyyaml>=6.0.0

# ML dependencies
sentence-transformers>=2.3.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Optional ML backends (install as needed)
# xgboost>=2.0.0
# lightgbm>=4.0.0

# Async support
anyio>=4.0.0

# CLI
click>=8.1.0
rich>=13.0.0
'

    # -----------------------------------------------------------------------
    # Example scripts
    # -----------------------------------------------------------------------
    _write_if_missing "examples/basic_fingerprint.py" '"""
Basic fingerprinting example.

Usage:
    python examples/basic_fingerprint.py
"""

import asyncio
from fingerprint.config import load_config
from fingerprint.core.analyzer import StructureAnalyzer


async def main():
    config = load_config()
    analyzer = StructureAnalyzer(config)

    # Analyze a URL
    structure = await analyzer.analyze_url("https://example.com/article")
    print(f"Page type: {structure.page_type}")
    print(f"Classes: {len(structure.css_class_map)}")

    # Compare with stored version
    changes = await analyzer.compare_with_stored("https://example.com/article")
    print(f"Similarity: {changes.similarity:.3f}")
    print(f"Breaking: {changes.breaking}")


if __name__ == "__main__":
    asyncio.run(main())
'

    _write_if_missing "examples/adaptive_mode.py" '"""
Adaptive mode example.

Usage:
    python examples/adaptive_mode.py
"""

import asyncio
from fingerprint.config import load_config
from fingerprint.core.analyzer import StructureAnalyzer


async def main():
    config = load_config()
    config.mode = "adaptive"

    analyzer = StructureAnalyzer(config)

    # Adaptive mode automatically selects best approach
    result = await analyzer.compare_with_stored("https://example.com")

    print(f"Mode used: {result.mode_used.value}")
    if result.escalated:
        print("Escalated to ML because:")
        for trigger in result.escalation_triggers:
            print(f"  - {trigger.name}: {trigger.reason}")


if __name__ == "__main__":
    asyncio.run(main())
'

    _write_if_missing "examples/ml_fingerprint.py" '"""
ML mode with Ollama Cloud example.

Usage:
    export OLLAMA_CLOUD_API_KEY="your-api-key"
    python examples/ml_fingerprint.py
"""

import asyncio
from fingerprint.config import load_config
from fingerprint.core.analyzer import StructureAnalyzer


async def main():
    config = load_config()
    config.ollama_cloud.enabled = True

    analyzer = StructureAnalyzer(config)

    structure = await analyzer.analyze_url("https://example.com")
    description = await analyzer.generate_description(structure)

    print("LLM Description:")
    print(description)


if __name__ == "__main__":
    asyncio.run(main())
'

    log_ok "Project scaffolding complete!"
    echo ""
    echo "  Next steps:"
    echo "    ./skills.sh config     # Generate configuration files"
    echo "    ./skills.sh deps       # Install dependencies"
    echo "    ./skills.sh info       # View architecture summary"
}

# ===========================================================================
# deps - Install dependencies into a virtual environment
# ===========================================================================
cmd_deps() {
    log_step "Setting up Python virtual environment and installing dependencies"

    # Check Python version
    local py_cmd=""
    for cmd in python3.11 python3.12 python3.13 python3; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
                py_cmd="$cmd"
                break
            fi
        fi
    done

    if [ -z "$py_cmd" ]; then
        log_error "Python 3.11+ is required but not found."
        exit 1
    fi

    log_info "Using $py_cmd ($($py_cmd --version 2>&1))"

    # Create virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment at ${VENV_DIR}"
        "$py_cmd" -m venv "$VENV_DIR"
    else
        log_info "Virtual environment already exists at ${VENV_DIR}"
    fi

    # Upgrade pip
    log_info "Upgrading pip"
    "$PIP" install --upgrade pip --quiet

    # Install package in editable mode with dev dependencies
    log_info "Installing adaptive-fingerprint in editable mode with dev dependencies"
    "$PIP" install -e "${PROJECT_ROOT}[dev]" --quiet 2>&1 | tail -5

    log_ok "Dependencies installed successfully!"
    echo ""
    echo "  Activate the environment with:"
    echo "    source ${VENV_DIR}/bin/activate"
}

# ===========================================================================
# config - Generate configuration files
# ===========================================================================
cmd_config() {
    log_step "Generating configuration files"

    # -----------------------------------------------------------------------
    # config.example.yaml
    # -----------------------------------------------------------------------
    _write_if_missing "config.example.yaml" '# Adaptive Fingerprint Configuration

# Fingerprinting mode: "rules", "ml", or "adaptive"
fingerprinting:
  mode: adaptive

  # Adaptive mode settings
  adaptive:
    class_change_threshold: 0.15      # Trigger ML if >15% classes changed
    rules_uncertainty_threshold: 0.80  # Trigger ML if rules similarity < 0.80
    cache_ml_results: true

  # Change classification thresholds
  thresholds:
    cosmetic: 0.95    # > 0.95 = cosmetic change
    minor: 0.85       # 0.85-0.95 = minor change
    moderate: 0.70    # 0.70-0.85 = moderate change
    breaking: 0.70    # < 0.70 = breaking change

# Ollama Cloud LLM settings
ollama_cloud:
  enabled: true
  model: "gemma3:12b"
  timeout: 30
  max_retries: 3
  temperature: 0.3
  max_tokens: 500

# Embedding model settings
embeddings:
  model: "all-MiniLM-L6-v2"
  cache_embeddings: true

# Redis storage
redis:
  url: "redis://localhost:6379/0"
  key_prefix: "fingerprint"
  ttl_seconds: 604800  # 7 days
  max_versions: 10

# HTTP fetching
http:
  user_agent: "AdaptiveFingerprint/1.0 (+https://example.com/bot-info)"
  timeout: 30
  max_retries: 3

# Compliance settings (ethical web access)
compliance:
  # robots.txt (RFC 9309)
  robots_txt:
    enabled: true
    cache_ttl: 3600           # Cache robots.txt for 1 hour
    respect_crawl_delay: true
    default_crawl_delay: 1.0  # Seconds between requests if not specified

  # Adaptive rate limiting
  rate_limiting:
    enabled: true
    default_delay: 1.0        # Default delay between requests (seconds)
    min_delay: 0.5            # Minimum delay
    max_delay: 30.0           # Maximum delay
    backoff_multiplier: 2.0   # Backoff on 429/503 responses
    adapt_to_response_time: true  # Slow down if server is slow

  # Anti-bot respect
  anti_bot:
    enabled: true
    respect_retry_after: true
    stop_on_captcha: true
    stop_on_block_page: true

# Legal compliance
legal:
  # CFAA (Computer Fraud and Abuse Act)
  cfaa:
    enabled: true
    require_public_access: true      # Only access publicly available pages
    block_authenticated_areas: true  # Do not access login-protected content
    block_api_endpoints: true        # Do not access API endpoints without permission

  # Terms of Service
  tos:
    enabled: true
    check_meta_tags: true     # Check for meta robots tags
    respect_noindex: true     # Respect noindex directives
    respect_nofollow: true    # Respect nofollow directives

  # GDPR (General Data Protection Regulation)
  gdpr:
    enabled: true
    pii_detection: true       # Detect personally identifiable information
    pii_handling: "redact"    # "redact", "pseudonymize", or "skip"
    log_pii_access: true      # Log when PII is encountered

  # CCPA (California Consumer Privacy Act)
  ccpa:
    enabled: true
    respect_opt_out: true     # Respect "do not sell" signals
    respect_gpc: true         # Respect Global Privacy Control header

# Content extraction
extraction:
  enabled: true
  output_dir: "./extracted"           # Directory for extracted content
  formats: ["json", "csv"]            # Output formats
  include_metadata: true              # Include extraction metadata
  include_html: false                 # Include raw HTML in output
  max_content_length: 1000000         # Max content size (bytes)

# Change alerting and review
alerting:
  enabled: true

  # Alert thresholds
  alert_on_breaking: true             # Alert on breaking changes
  alert_on_moderate: false            # Alert on moderate changes
  alert_threshold: 0.70               # Similarity below this triggers alert

  # Review queue
  review_queue:
    enabled: true
    auto_approve_cosmetic: true       # Auto-approve cosmetic changes
    auto_approve_minor: false         # Auto-approve minor changes
    require_review_breaking: true     # Require manual review for breaking
    max_queue_size: 1000              # Max pending reviews

  # Notifications
  notifications:
    log: true                         # Log alerts
    webhook:
      enabled: false
      url: ""                         # Webhook URL for alerts
    email:
      enabled: false
      smtp_host: ""
      smtp_port: 587
      recipients: []

# Verbose logging
verbose:
  enabled: true
  level: 2  # 0=errors, 1=warnings, 2=info, 3=debug
  format: "structured"  # "structured" or "plain"
  include_timestamp: true
'

    # -----------------------------------------------------------------------
    # .env.example
    # -----------------------------------------------------------------------
    _write_if_missing ".env.example" '# Required for Ollama Cloud LLM integration
OLLAMA_CLOUD_API_KEY=your-api-key-here

# Optional overrides
FINGERPRINT_MODE=adaptive
FINGERPRINT_VERBOSE=2
REDIS_URL=redis://localhost:6379/0
'

    # Copy example to actual config if not present
    if [ ! -f "${PROJECT_ROOT}/config.yaml" ]; then
        cp "${PROJECT_ROOT}/config.example.yaml" "${PROJECT_ROOT}/config.yaml"
        log_info "Copied config.example.yaml -> config.yaml (edit as needed)"
    fi

    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
        log_info "Copied .env.example -> .env (set your OLLAMA_CLOUD_API_KEY)"
    fi

    log_ok "Configuration files generated!"
}

# ===========================================================================
# redis - Start Redis container
# ===========================================================================
cmd_redis() {
    log_step "Starting Redis for structure storage"

    if ! command -v docker &>/dev/null; then
        log_error "Docker is not installed. Please install Docker or provide a Redis instance."
        echo ""
        echo "  Manual Redis setup:"
        echo "    export REDIS_URL=redis://your-redis-host:6379/0"
        exit 1
    fi

    # Check if container already running
    if docker ps --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER_NAME}$"; then
        log_info "Redis container '${REDIS_CONTAINER_NAME}' is already running"
        echo "  Redis URL: redis://localhost:${REDIS_PORT}/0"
        return
    fi

    # Check if container exists but stopped
    if docker ps -a --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER_NAME}$"; then
        log_info "Starting existing Redis container"
        docker start "${REDIS_CONTAINER_NAME}" >/dev/null
    else
        log_info "Creating and starting Redis container"
        docker run -d \
            --name "${REDIS_CONTAINER_NAME}" \
            -p "${REDIS_PORT}:6379" \
            redis:7-alpine >/dev/null
    fi

    # Wait for Redis to be ready
    local retries=10
    while [ $retries -gt 0 ]; do
        if docker exec "${REDIS_CONTAINER_NAME}" redis-cli ping 2>/dev/null | grep -q PONG; then
            break
        fi
        retries=$((retries - 1))
        sleep 0.5
    done

    if [ $retries -eq 0 ]; then
        log_error "Redis failed to start"
        exit 1
    fi

    log_ok "Redis is running"
    echo "  Container: ${REDIS_CONTAINER_NAME}"
    echo "  URL:       redis://localhost:${REDIS_PORT}/0"
}

# ===========================================================================
# redis-stop - Stop Redis container
# ===========================================================================
cmd_redis_stop() {
    log_step "Stopping Redis container"

    if ! command -v docker &>/dev/null; then
        log_warn "Docker is not installed"
        return
    fi

    if docker ps --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER_NAME}$"; then
        docker stop "${REDIS_CONTAINER_NAME}" >/dev/null
        log_ok "Redis container stopped"
    else
        log_info "Redis container is not running"
    fi
}

# ===========================================================================
# lint - Run linting
# ===========================================================================
cmd_lint() {
    log_step "Running linter (ruff)"

    _ensure_venv

    if ! "${VENV_DIR}/bin/ruff" --version &>/dev/null 2>&1; then
        log_info "Installing ruff..."
        "$PIP" install ruff --quiet
    fi

    "${VENV_DIR}/bin/ruff" check "${PROJECT_ROOT}/fingerprint" "$@"
    log_ok "Linting passed"
}

# ===========================================================================
# test - Run the test suite
# ===========================================================================
cmd_test() {
    log_step "Running test suite"

    _ensure_venv

    if ! "${VENV_DIR}/bin/pytest" --version &>/dev/null 2>&1; then
        log_info "Installing pytest..."
        "$PIP" install pytest pytest-asyncio pytest-cov --quiet
    fi

    "${VENV_DIR}/bin/pytest" "${PROJECT_ROOT}/tests" \
        --tb=short \
        -v \
        "$@"
}

# ===========================================================================
# clean - Remove generated artifacts
# ===========================================================================
cmd_clean() {
    log_step "Cleaning generated artifacts"

    # __pycache__
    find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    log_info "Removed __pycache__ directories"

    # .pytest_cache
    rm -rf "${PROJECT_ROOT}/.pytest_cache"
    log_info "Removed .pytest_cache"

    # egg-info
    rm -rf "${PROJECT_ROOT}"/*.egg-info
    rm -rf "${PROJECT_ROOT}"/fingerprint/*.egg-info
    log_info "Removed egg-info directories"

    # extracted output
    if [ -d "${PROJECT_ROOT}/extracted" ]; then
        rm -rf "${PROJECT_ROOT}/extracted"
        log_info "Removed extracted/ output directory"
    fi

    # Optional: remove venv
    if [ "${1:-}" = "--all" ]; then
        if [ -d "$VENV_DIR" ]; then
            rm -rf "$VENV_DIR"
            log_info "Removed virtual environment"
        fi
    fi

    log_ok "Clean complete"
}

# ===========================================================================
# info - Display project architecture summary
# ===========================================================================
cmd_info() {
    echo ""
    echo -e "${BOLD}Adaptive Structure Fingerprinting System${NC}"
    echo -e "${BOLD}=========================================${NC}"
    echo ""
    echo "An intelligent web structure fingerprinting system with adaptive"
    echo "learning, Ollama Cloud LLM integration, ethical compliance,"
    echo "and comprehensive verbose logging."
    echo ""
    echo -e "${BOLD}Fingerprinting Modes:${NC}"
    echo "  rules     ~15ms   Fast, deterministic DOM structure analysis"
    echo "  ml        ~200ms  Semantic similarity via sentence-transformers"
    echo "  adaptive  ~15-200ms  Smart selection: rules first, ML on escalation"
    echo ""
    echo -e "${BOLD}Escalation Triggers (adaptive mode):${NC}"
    echo "  CLASS_VOLATILITY   >15% of CSS classes changed"
    echo "  RULES_UNCERTAINTY  Rules similarity < 0.80"
    echo "  KNOWN_VOLATILE     Domain flagged as volatile in history"
    echo "  RENAME_PATTERN     Detected CSS class rename patterns"
    echo ""
    echo -e "${BOLD}Change Classification Thresholds:${NC}"
    echo "  cosmetic   > 0.95  similarity"
    echo "  minor      0.85 - 0.95"
    echo "  moderate   0.70 - 0.85"
    echo "  breaking   < 0.70"
    echo ""
    echo -e "${BOLD}Compliance Pipeline (mandatory for all fetches):${NC}"
    echo "  1. CFAA Check        Is access authorized?"
    echo "  2. ToS Check         Does ToS allow crawling?"
    echo "  3. robots.txt        Is path allowed? (RFC 9309)"
    echo "  4. Rate Limiter      Acquire slot, respect Crawl-delay"
    echo "  5. HTTP Fetch        Make request with proper headers"
    echo "  6. Anti-Bot Check    Detect captcha/block pages"
    echo "  7. GDPR/CCPA Check   Scan for PII, apply handling"
    echo ""
    echo -e "${BOLD}Package Structure:${NC}"
    echo "  fingerprint/"
    echo "    __main__.py         CLI entry point"
    echo "    config.py           Configuration (Pydantic settings + YAML)"
    echo "    models.py           Core data models (dataclasses)"
    echo "    exceptions.py       Custom exception hierarchy"
    echo "    core/               Orchestrator, fetcher, verbose logging"
    echo "    adaptive/           Rules-based DOM analysis, change detection"
    echo "    ml/                 Embeddings, Ollama Cloud, classifier"
    echo "    storage/            Redis: structures, embeddings, review queue"
    echo "    compliance/         robots.txt (RFC 9309), rate limiter, bot detector"
    echo "    legal/              CFAA, ToS, GDPR, CCPA"
    echo "    extraction/         Content extractor, file writer, format handlers"
    echo "    alerting/           Change monitor, review queue, notifiers"
    echo "    utils/              URL and HTML utilities"
    echo ""
    echo -e "${BOLD}External Services:${NC}"
    echo "  Redis           Structure storage, versioning, review queue"
    echo "  Ollama Cloud    LLM descriptions (gemma3:12b default)"
    echo ""
    echo -e "${BOLD}Verbose Logging Modules:${NC}"
    echo "  ANALYZER  STRUCTURE  CHANGE   ADAPTIVE  ML       OLLAMA"
    echo "  STORE     ROBOTS     RATELIMIT ANTIBOT  CFAA     TOS"
    echo "  GDPR      CCPA       EXTRACT  FILEWRITER ALERT   REVIEW"
    echo "  NOTIFY    CLASSIFY   CACHE    FETCH     EMBED_STORE"
    echo ""
    echo -e "${BOLD}CLI Commands:${NC}"
    echo "  fingerprint analyze   --url <URL>              Analyze structure"
    echo "  fingerprint compare   --url <URL> --mode <M>   Compare with stored"
    echo "  fingerprint describe  --url <URL>              Generate LLM description"
    echo "  fingerprint extract   --url <URL> --format F   Extract content"
    echo "  fingerprint review    list|approve|reject|stats Review queue"
    echo ""
    echo -e "${BOLD}Redis Key Schema:${NC}"
    echo "  {prefix}:structure:{domain}:{page_type}:{variant}      Current"
    echo "  {prefix}:structure:{domain}:{page_type}:{variant}:v{n} Version"
    echo "  {prefix}:embedding:{domain}:{page_type}:{variant}      Embedding"
    echo "  {prefix}:volatile:{domain}                              Volatile flag"
    echo "  {prefix}:review:pending                                 Queue"
    echo "  {prefix}:review:item:{id}                               Item data"
    echo ""
}

# ===========================================================================
# doctor - Verify prerequisites
# ===========================================================================
cmd_doctor() {
    log_step "Checking prerequisites"
    local all_ok=true

    # Python
    echo -n "  Python 3.11+ ... "
    local py_found=false
    for cmd in python3.11 python3.12 python3.13 python3; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
                echo -e "${GREEN}OK${NC} ($cmd $ver)"
                py_found=true
                break
            fi
        fi
    done
    if [ "$py_found" = false ]; then
        echo -e "${RED}MISSING${NC}"
        all_ok=false
    fi

    # pip
    echo -n "  pip ... "
    if command -v pip3 &>/dev/null || command -v pip &>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}MISSING${NC}"
        all_ok=false
    fi

    # Docker (optional)
    echo -n "  Docker ... "
    if command -v docker &>/dev/null; then
        echo -e "${GREEN}OK${NC} ($(docker --version 2>/dev/null | head -1))"
    else
        echo -e "${YELLOW}NOT FOUND${NC} (optional - needed for Redis container)"
    fi

    # Redis (check connectivity)
    echo -n "  Redis ... "
    if command -v redis-cli &>/dev/null; then
        if redis-cli -p "${REDIS_PORT}" ping 2>/dev/null | grep -q PONG; then
            echo -e "${GREEN}OK${NC} (localhost:${REDIS_PORT})"
        else
            echo -e "${YELLOW}NOT RUNNING${NC} (run: ./skills.sh redis)"
        fi
    elif docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${REDIS_CONTAINER_NAME}$"; then
        echo -e "${GREEN}OK${NC} (Docker container)"
    else
        echo -e "${YELLOW}NOT RUNNING${NC} (run: ./skills.sh redis)"
    fi

    # Git
    echo -n "  Git ... "
    if command -v git &>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}MISSING${NC}"
        all_ok=false
    fi

    # Virtual environment
    echo -n "  Virtual env ... "
    if [ -d "$VENV_DIR" ]; then
        echo -e "${GREEN}OK${NC} (${VENV_DIR})"
    else
        echo -e "${YELLOW}NOT CREATED${NC} (run: ./skills.sh deps)"
    fi

    # Config files
    echo -n "  config.yaml ... "
    if [ -f "${PROJECT_ROOT}/config.yaml" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}MISSING${NC} (run: ./skills.sh config)"
    fi

    echo -n "  .env ... "
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}MISSING${NC} (run: ./skills.sh config)"
    fi

    echo ""
    if [ "$all_ok" = true ]; then
        log_ok "All required prerequisites are available"
    else
        log_error "Some required prerequisites are missing"
        exit 1
    fi
}

# ===========================================================================
# help - Show usage
# ===========================================================================
cmd_help() {
    echo ""
    echo -e "${BOLD}Adaptive Fingerprint - skills.sh${NC}"
    echo ""
    echo "Usage: ./skills.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  scaffold      Create full project directory structure and source stubs"
    echo "  deps          Create venv and install Python dependencies"
    echo "  config        Generate config.example.yaml, .env.example, config.yaml, .env"
    echo "  redis         Start Redis via Docker (port ${REDIS_PORT})"
    echo "  redis-stop    Stop the Redis Docker container"
    echo "  lint          Run ruff linter on the fingerprint package"
    echo "  test          Run pytest test suite"
    echo "  clean         Remove __pycache__, .pytest_cache, egg-info, extracted/"
    echo "  clean --all   Also remove the virtual environment"
    echo "  info          Display project architecture and module summary"
    echo "  doctor        Verify all prerequisites (Python, Docker, Redis, etc.)"
    echo "  help          Show this help message"
    echo ""
    echo "Quick start:"
    echo "  ./skills.sh doctor       # Check prerequisites"
    echo "  ./skills.sh scaffold     # Create project structure"
    echo "  ./skills.sh config       # Generate config files"
    echo "  ./skills.sh deps         # Install dependencies"
    echo "  ./skills.sh redis        # Start Redis"
    echo "  source .venv/bin/activate"
    echo "  fingerprint analyze --url https://example.com"
    echo ""
}

# ===========================================================================
# Utility functions
# ===========================================================================

# Write file only if it does not already exist
_write_if_missing() {
    local filepath="${PROJECT_ROOT}/$1"
    local content="$2"

    if [ ! -f "$filepath" ]; then
        mkdir -p "$(dirname "$filepath")"
        printf '%s' "$content" > "$filepath"
        log_info "Created: $1"
    else
        log_info "Exists:  $1 (skipped)"
    fi
}

# Ensure virtual environment exists
_ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found. Run: ./skills.sh deps"
        exit 1
    fi
}

# ===========================================================================
# Main dispatcher
# ===========================================================================
main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        scaffold)   cmd_scaffold "$@" ;;
        deps)       cmd_deps "$@" ;;
        config)     cmd_config "$@" ;;
        redis)      cmd_redis "$@" ;;
        redis-stop) cmd_redis_stop "$@" ;;
        lint)       cmd_lint "$@" ;;
        test)       cmd_test "$@" ;;
        clean)      cmd_clean "$@" ;;
        info)       cmd_info "$@" ;;
        doctor)     cmd_doctor "$@" ;;
        help|--help|-h)
                    cmd_help "$@" ;;
        *)
            log_error "Unknown command: $cmd"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
