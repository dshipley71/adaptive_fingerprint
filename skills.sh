#!/usr/bin/env bash
# =============================================================================
# skills.sh - Adaptive Structure Fingerprinting System
# =============================================================================
#
# Agent Skills manager and project development utilities.
#
# This script serves two roles:
#   1. Agent Skills discovery, listing, and validation (SKILL.md format per
#      the Anthropic Agent Skills specification: https://skills.sh/)
#   2. Project scaffolding, dependency management, and dev workflow commands
#
# Usage:
#   chmod +x skills.sh
#   ./skills.sh <command>
#
# Skills commands:
#   skills              - List all available Agent Skills
#   skills show <name>  - Display a skill's SKILL.md content
#   skills validate     - Validate all skills against the Agent Skills spec
#   skills tree         - Show the full skills directory tree
#
# Project commands:
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
SKILLS_DIR="${PROJECT_ROOT}/skills"
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
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BOLD}==> $*${NC}"; }

# ===========================================================================
# skills - Agent Skills management
# ===========================================================================
cmd_skills() {
    local subcmd="${1:-list}"
    shift || true

    case "$subcmd" in
        list)       _skills_list ;;
        show)       _skills_show "$@" ;;
        validate)   _skills_validate ;;
        tree)       _skills_tree ;;
        *)
            # If the argument looks like a skill name, try show
            if [ -d "${SKILLS_DIR}/${subcmd}" ]; then
                _skills_show "$subcmd"
            else
                log_error "Unknown skills subcommand: $subcmd"
                echo "  Usage: ./skills.sh skills [list|show <name>|validate|tree]"
                exit 1
            fi
            ;;
    esac
}

_skills_list() {
    echo ""
    echo -e "${BOLD}Adaptive Fingerprint - Agent Skills${NC}"
    echo -e "${DIM}Format: Anthropic Agent Skills (SKILL.md)${NC}"
    echo -e "${DIM}Spec:   https://skills.sh/${NC}"
    echo ""

    if [ ! -d "$SKILLS_DIR" ]; then
        log_warn "No skills directory found at ${SKILLS_DIR}"
        return
    fi

    local count=0

    for skill_dir in "${SKILLS_DIR}"/*/; do
        [ -d "$skill_dir" ] || continue
        local skill_md="${skill_dir}SKILL.md"
        [ -f "$skill_md" ] || continue

        local name description
        name=$(_extract_frontmatter_field "$skill_md" "name")
        description=$(_extract_frontmatter_field "$skill_md" "description")

        # Truncate description for display
        if [ ${#description} -gt 100 ]; then
            description="${description:0:97}..."
        fi

        printf "  ${GREEN}%-28s${NC} %s\n" "$name" "$description"
        count=$((count + 1))
    done

    echo ""
    echo -e "  ${DIM}${count} skills available${NC}"
    echo ""
    echo "  Show a skill:     ./skills.sh skills show <name>"
    echo "  Validate all:     ./skills.sh skills validate"
    echo "  Directory tree:   ./skills.sh skills tree"
    echo ""
}

_skills_show() {
    local name="${1:-}"
    if [ -z "$name" ]; then
        log_error "Skill name required"
        echo "  Usage: ./skills.sh skills show <name>"
        echo ""
        echo "  Available skills:"
        for skill_dir in "${SKILLS_DIR}"/*/; do
            [ -d "$skill_dir" ] || continue
            [ -f "${skill_dir}SKILL.md" ] || continue
            local sname
            sname=$(_extract_frontmatter_field "${skill_dir}SKILL.md" "name")
            echo "    $sname"
        done
        exit 1
    fi

    local skill_md="${SKILLS_DIR}/${name}/SKILL.md"
    if [ ! -f "$skill_md" ]; then
        log_error "Skill not found: $name"
        echo "  Looking for: ${skill_md}"
        exit 1
    fi

    echo ""
    echo -e "${BOLD}Skill: ${name}${NC}"
    echo -e "${DIM}$(printf '%.0s-' {1..60})${NC}"

    # Show metadata
    local desc
    desc=$(_extract_frontmatter_field "$skill_md" "description")
    echo -e "${CYAN}Description:${NC} ${desc}"

    # List references
    local ref_dir="${SKILLS_DIR}/${name}/references"
    if [ -d "$ref_dir" ] && [ "$(ls -A "$ref_dir" 2>/dev/null)" ]; then
        echo -e "${CYAN}References:${NC}"
        for ref in "${ref_dir}"/*; do
            [ -f "$ref" ] || continue
            echo "  - $(basename "$ref")"
        done
    fi

    # List scripts
    local script_dir="${SKILLS_DIR}/${name}/scripts"
    if [ -d "$script_dir" ] && [ "$(ls -A "$script_dir" 2>/dev/null)" ]; then
        echo -e "${CYAN}Scripts:${NC}"
        for scr in "${script_dir}"/*; do
            [ -f "$scr" ] || continue
            echo "  - $(basename "$scr")"
        done
    fi

    echo -e "${DIM}$(printf '%.0s-' {1..60})${NC}"
    echo ""

    # Display SKILL.md body (after frontmatter)
    awk 'BEGIN{fm=0} /^---$/{fm++; next} fm>=2{print}' "$skill_md"
    echo ""
}

_skills_validate() {
    log_step "Validating Agent Skills"

    if [ ! -d "$SKILLS_DIR" ]; then
        log_error "No skills directory found at ${SKILLS_DIR}"
        exit 1
    fi

    local total=0
    local passed=0
    local failed=0
    local errors=""

    for skill_dir in "${SKILLS_DIR}"/*/; do
        [ -d "$skill_dir" ] || continue
        local skill_name
        skill_name=$(basename "$skill_dir")
        local skill_md="${skill_dir}SKILL.md"
        total=$((total + 1))

        local skill_errors=""

        # Check SKILL.md exists
        if [ ! -f "$skill_md" ]; then
            skill_errors="${skill_errors}\n    - Missing SKILL.md"
        else
            # Check frontmatter delimiters
            local first_line
            first_line=$(head -1 "$skill_md")
            if [ "$first_line" != "---" ]; then
                skill_errors="${skill_errors}\n    - SKILL.md must begin with --- (YAML frontmatter)"
            fi

            # Check required fields
            local name_val desc_val
            name_val=$(_extract_frontmatter_field "$skill_md" "name")
            desc_val=$(_extract_frontmatter_field "$skill_md" "description")

            if [ -z "$name_val" ]; then
                skill_errors="${skill_errors}\n    - Missing required field: name"
            fi

            if [ -z "$desc_val" ]; then
                skill_errors="${skill_errors}\n    - Missing required field: description"
            elif [ ${#desc_val} -gt 1024 ]; then
                skill_errors="${skill_errors}\n    - Description exceeds 1024 characters (${#desc_val})"
            fi

            # Check for markdown body after frontmatter
            local body_lines
            body_lines=$(awk 'BEGIN{fm=0} /^---$/{fm++; next} fm>=2{print}' "$skill_md" | wc -l)
            if [ "$body_lines" -lt 3 ]; then
                skill_errors="${skill_errors}\n    - SKILL.md body has very few lines (${body_lines}); add instructions"
            fi

            # Check for disallowed frontmatter fields
            local fm_fields
            fm_fields=$(awk 'BEGIN{fm=0} /^---$/{fm++; next} fm==1 && /^[a-z]/{print $1}' "$skill_md" | sed 's/://')
            for fld in $fm_fields; do
                case "$fld" in
                    name|description|license|allowed-tools|metadata) ;;
                    *) skill_errors="${skill_errors}\n    - Unknown frontmatter field: ${fld}" ;;
                esac
            done

            # Check naming convention (should match directory name)
            if [ -n "$name_val" ] && [ "$name_val" != "$skill_name" ]; then
                skill_errors="${skill_errors}\n    - Name '${name_val}' does not match directory '${skill_name}'"
            fi
        fi

        if [ -z "$skill_errors" ]; then
            printf "  ${GREEN}PASS${NC}  %s\n" "$skill_name"
            passed=$((passed + 1))
        else
            printf "  ${RED}FAIL${NC}  %s\n" "$skill_name"
            echo -e "$skill_errors"
            failed=$((failed + 1))
            errors="${errors}\n  ${skill_name}:${skill_errors}"
        fi
    done

    echo ""
    echo -e "  Total: ${total}  ${GREEN}Passed: ${passed}${NC}  ${RED}Failed: ${failed}${NC}"

    if [ "$failed" -gt 0 ]; then
        echo ""
        log_error "Validation failed for ${failed} skill(s)"
        exit 1
    else
        echo ""
        log_ok "All skills passed validation"
    fi
}

_skills_tree() {
    log_step "Skills directory structure"
    echo ""

    if [ ! -d "$SKILLS_DIR" ]; then
        log_warn "No skills directory found"
        return
    fi

    echo -e "${BOLD}skills/${NC}"
    for skill_dir in "${SKILLS_DIR}"/*/; do
        [ -d "$skill_dir" ] || continue
        local skill_name
        skill_name=$(basename "$skill_dir")
        local skill_md="${skill_dir}SKILL.md"
        local desc=""
        if [ -f "$skill_md" ]; then
            desc=$(_extract_frontmatter_field "$skill_md" "name")
        fi

        echo -e "  ${GREEN}${skill_name}/${NC}"
        # List files
        for item in "${skill_dir}"*; do
            [ -e "$item" ] || continue
            local bname
            bname=$(basename "$item")
            if [ -f "$item" ]; then
                echo "    ${bname}"
            elif [ -d "$item" ]; then
                echo -e "    ${BLUE}${bname}/${NC}"
                for sub in "$item"/*; do
                    [ -e "$sub" ] || continue
                    echo "      $(basename "$sub")"
                done
            fi
        done
    done
    echo ""
}

# Extract a YAML frontmatter field value from a SKILL.md file
_extract_frontmatter_field() {
    local file="$1"
    local field="$2"

    # Extract frontmatter block and find the field
    awk -v field="$field" '
    BEGIN { in_fm=0; found="" }
    /^---$/ { in_fm++; next }
    in_fm == 1 {
        # Match field: value (possibly multiline, we take first line)
        if ($0 ~ "^" field ":") {
            sub("^" field ":[[:space:]]*", "")
            found = $0
        }
    }
    in_fm >= 2 { exit }
    END { print found }
    ' "$file"
}

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
    # Source file stubs
    # -----------------------------------------------------------------------
    _write_if_missing "fingerprint/models.py" '"""Core data models (dataclasses and enums)."""
'
    _write_if_missing "fingerprint/config.py" '"""Configuration management using Pydantic settings and YAML."""
'
    _write_if_missing "fingerprint/exceptions.py" '"""Custom exception hierarchy (all inherit from FingerprintError)."""
'
    _write_if_missing "fingerprint/__main__.py" '"""CLI entry point (click-based)."""
'
    _write_if_missing "fingerprint/core/analyzer.py" '"""Main analyzer orchestrator (StructureAnalyzer)."""
'
    _write_if_missing "fingerprint/core/fetcher.py" '"""HTTP fetcher with compliance pipeline (ComplianceFetcher, HTTPFetcher)."""
'
    _write_if_missing "fingerprint/core/verbose.py" '"""Verbose logging system (VerboseLogger, get_logger, set_logger)."""
'
    _write_if_missing "fingerprint/adaptive/structure_analyzer.py" '"""Rules-based DOM structure analysis (DOMStructureAnalyzer)."""
'
    _write_if_missing "fingerprint/adaptive/change_detector.py" '"""Change detection and classification (ChangeDetector)."""
'
    _write_if_missing "fingerprint/adaptive/strategy_learner.py" '"""CSS selector inference for extraction (StrategyLearner)."""
'
    _write_if_missing "fingerprint/ml/embeddings.py" '"""Embedding generation via sentence-transformers (EmbeddingGenerator)."""
'
    _write_if_missing "fingerprint/ml/ollama_client.py" '"""Ollama Cloud API client (OllamaCloudClient)."""
'
    _write_if_missing "fingerprint/ml/classifier.py" '"""Page type classification using embeddings (PageClassifier)."""
'
    _write_if_missing "fingerprint/storage/structure_store.py" '"""Redis storage for page structures (StructureStore)."""
'
    _write_if_missing "fingerprint/storage/embedding_store.py" '"""Redis storage for embeddings (EmbeddingStore)."""
'
    _write_if_missing "fingerprint/storage/review_store.py" '"""Redis storage for review queue (ReviewStore)."""
'
    _write_if_missing "fingerprint/storage/cache.py" '"""In-memory TTL-based cache (Cache[T])."""
'
    _write_if_missing "fingerprint/compliance/robots_parser.py" '"""robots.txt parser - RFC 9309 (RobotsParser, RobotsChecker)."""
'
    _write_if_missing "fingerprint/compliance/rate_limiter.py" '"""Adaptive per-domain rate limiter (RateLimiter, DomainState)."""
'
    _write_if_missing "fingerprint/compliance/bot_detector.py" '"""Anti-bot detection and respect (BotDetector, BotCheckResult)."""
'
    _write_if_missing "fingerprint/legal/cfaa_checker.py" '"""CFAA authorization checker (CFAAChecker, AuthorizationResult)."""
'
    _write_if_missing "fingerprint/legal/tos_checker.py" '"""Terms of Service compliance (ToSChecker, ToSResult)."""
'
    _write_if_missing "fingerprint/legal/gdpr_handler.py" '"""GDPR PII detection and handling (GDPRHandler, PIIDetectionResult)."""
'
    _write_if_missing "fingerprint/legal/ccpa_handler.py" '"""CCPA compliance handler (CCPAHandler, CCPACheckResult)."""
'
    _write_if_missing "fingerprint/extraction/extractor.py" '"""Content extraction engine (ContentExtractor)."""
'
    _write_if_missing "fingerprint/extraction/file_writer.py" '"""File writer for extracted content (FileWriter)."""
'
    _write_if_missing "fingerprint/extraction/formats.py" '"""Output format handlers: JSON, CSV, Markdown (OutputFormatter)."""
'
    _write_if_missing "fingerprint/alerting/change_monitor.py" '"""Monitor for breaking changes (ChangeMonitor)."""
'
    _write_if_missing "fingerprint/alerting/review_queue.py" '"""Manual review queue (ReviewQueue)."""
'
    _write_if_missing "fingerprint/alerting/notifiers.py" '"""Alert notifiers: log, webhook, email."""
'
    _write_if_missing "fingerprint/utils/url_utils.py" '"""URL normalization and domain extraction."""
'
    _write_if_missing "fingerprint/utils/html_utils.py" '"""HTML parsing utilities."""
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
    _write_if_missing "requirements.txt" '# Core
httpx>=0.27.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
redis>=5.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
pyyaml>=6.0.0

# ML
sentence-transformers>=2.3.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Async & CLI
anyio>=4.0.0
click>=8.1.0
rich>=13.0.0
'

    # -----------------------------------------------------------------------
    # Example scripts
    # -----------------------------------------------------------------------
    _write_if_missing "examples/basic_fingerprint.py" '"""Basic fingerprinting example."""

import asyncio
from fingerprint.config import load_config
from fingerprint.core.analyzer import StructureAnalyzer

async def main():
    config = load_config()
    analyzer = StructureAnalyzer(config)
    structure = await analyzer.analyze_url("https://example.com/article")
    print(f"Page type: {structure.page_type}")
    changes = await analyzer.compare_with_stored("https://example.com/article")
    print(f"Similarity: {changes.similarity:.3f}, Breaking: {changes.breaking}")

if __name__ == "__main__":
    asyncio.run(main())
'

    _write_if_missing "examples/adaptive_mode.py" '"""Adaptive mode example."""

import asyncio
from fingerprint.config import load_config
from fingerprint.core.analyzer import StructureAnalyzer

async def main():
    config = load_config()
    config.mode = "adaptive"
    analyzer = StructureAnalyzer(config)
    result = await analyzer.compare_with_stored("https://example.com")
    print(f"Mode used: {result.mode_used.value}")
    if result.escalated:
        for trigger in result.escalation_triggers:
            print(f"  Escalated: {trigger.name} - {trigger.reason}")

if __name__ == "__main__":
    asyncio.run(main())
'

    _write_if_missing "examples/ml_fingerprint.py" '"""ML mode with Ollama Cloud example (set OLLAMA_CLOUD_API_KEY)."""

import asyncio
from fingerprint.config import load_config
from fingerprint.core.analyzer import StructureAnalyzer

async def main():
    config = load_config()
    config.ollama_cloud.enabled = True
    analyzer = StructureAnalyzer(config)
    structure = await analyzer.analyze_url("https://example.com")
    description = await analyzer.generate_description(structure)
    print(description)

if __name__ == "__main__":
    asyncio.run(main())
'

    log_ok "Project scaffolding complete!"
    echo ""
    echo "  Next steps:"
    echo "    ./skills.sh config     # Generate configuration files"
    echo "    ./skills.sh deps       # Install dependencies"
    echo "    ./skills.sh skills     # View available Agent Skills"
}

# ===========================================================================
# deps - Install dependencies into a virtual environment
# ===========================================================================
cmd_deps() {
    log_step "Setting up Python virtual environment and installing dependencies"

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

    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment at ${VENV_DIR}"
        "$py_cmd" -m venv "$VENV_DIR"
    else
        log_info "Virtual environment already exists at ${VENV_DIR}"
    fi

    log_info "Upgrading pip"
    "$PIP" install --upgrade pip --quiet

    log_info "Installing adaptive-fingerprint in editable mode with dev dependencies"
    "$PIP" install -e "${PROJECT_ROOT}[dev]" --quiet 2>&1 | tail -5

    log_ok "Dependencies installed successfully!"
    echo "  Activate: source ${VENV_DIR}/bin/activate"
}

# ===========================================================================
# config - Generate configuration files
# ===========================================================================
cmd_config() {
    log_step "Generating configuration files"

    _write_if_missing "config.example.yaml" '# Adaptive Fingerprint Configuration
fingerprinting:
  mode: adaptive
  adaptive:
    class_change_threshold: 0.15
    rules_uncertainty_threshold: 0.80
    cache_ml_results: true
  thresholds:
    cosmetic: 0.95
    minor: 0.85
    moderate: 0.70
    breaking: 0.70

ollama_cloud:
  enabled: true
  model: "gemma3:12b"
  timeout: 30
  max_retries: 3
  temperature: 0.3
  max_tokens: 500

embeddings:
  model: "all-MiniLM-L6-v2"
  cache_embeddings: true

redis:
  url: "redis://localhost:6379/0"
  key_prefix: "fingerprint"
  ttl_seconds: 604800
  max_versions: 10

http:
  user_agent: "AdaptiveFingerprint/1.0 (+https://example.com/bot-info)"
  timeout: 30
  max_retries: 3

compliance:
  robots_txt:
    enabled: true
    cache_ttl: 3600
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
    pii_handling: "redact"
    log_pii_access: true
  ccpa:
    enabled: true
    respect_opt_out: true
    respect_gpc: true

extraction:
  enabled: true
  output_dir: "./extracted"
  formats: ["json", "csv"]
  include_metadata: true
  include_html: false
  max_content_length: 1000000

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

verbose:
  enabled: true
  level: 2
  format: "structured"
  include_timestamp: true
'

    _write_if_missing ".env.example" '# Required for Ollama Cloud LLM integration
OLLAMA_CLOUD_API_KEY=your-api-key-here

# Optional overrides
FINGERPRINT_MODE=adaptive
FINGERPRINT_VERBOSE=2
REDIS_URL=redis://localhost:6379/0
'

    if [ ! -f "${PROJECT_ROOT}/config.yaml" ]; then
        cp "${PROJECT_ROOT}/config.example.yaml" "${PROJECT_ROOT}/config.yaml"
        log_info "Copied config.example.yaml -> config.yaml"
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
        log_error "Docker is not installed. Install Docker or set REDIS_URL manually."
        exit 1
    fi

    if docker ps --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER_NAME}$"; then
        log_info "Redis container '${REDIS_CONTAINER_NAME}' is already running"
        echo "  Redis URL: redis://localhost:${REDIS_PORT}/0"
        return
    fi

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

    log_ok "Redis is running at redis://localhost:${REDIS_PORT}/0"
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
        "$PIP" install ruff --quiet
    fi

    "${VENV_DIR}/bin/ruff" check "${PROJECT_ROOT}/fingerprint" "$@"
    log_ok "Linting passed"
}

# ===========================================================================
# test - Run test suite
# ===========================================================================
cmd_test() {
    log_step "Running test suite"
    _ensure_venv

    if ! "${VENV_DIR}/bin/pytest" --version &>/dev/null 2>&1; then
        "$PIP" install pytest pytest-asyncio pytest-cov --quiet
    fi

    "${VENV_DIR}/bin/pytest" "${PROJECT_ROOT}/tests" --tb=short -v "$@"
}

# ===========================================================================
# clean - Remove generated artifacts
# ===========================================================================
cmd_clean() {
    log_step "Cleaning generated artifacts"

    find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    rm -rf "${PROJECT_ROOT}/.pytest_cache"
    rm -rf "${PROJECT_ROOT}"/*.egg-info
    rm -rf "${PROJECT_ROOT}"/fingerprint/*.egg-info
    [ -d "${PROJECT_ROOT}/extracted" ] && rm -rf "${PROJECT_ROOT}/extracted"

    if [ "${1:-}" = "--all" ] && [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        log_info "Removed virtual environment"
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
    echo "Intelligent web structure fingerprinting with adaptive learning,"
    echo "Ollama Cloud LLM integration, ethical compliance, and verbose logging."
    echo ""
    echo -e "${BOLD}Fingerprinting Modes:${NC}"
    echo "  rules     ~15ms     Deterministic DOM structure analysis"
    echo "  ml        ~200ms    Semantic similarity via sentence-transformers"
    echo "  adaptive  ~15-200ms Smart: rules first, ML on escalation"
    echo ""
    echo -e "${BOLD}Escalation Triggers (adaptive mode):${NC}"
    echo "  CLASS_VOLATILITY   >15% of CSS classes changed"
    echo "  RULES_UNCERTAINTY  Rules similarity < 0.80"
    echo "  KNOWN_VOLATILE     Domain flagged volatile"
    echo "  RENAME_PATTERN     CSS class rename patterns"
    echo ""
    echo -e "${BOLD}Change Thresholds:${NC}"
    echo "  cosmetic > 0.95 | minor 0.85-0.95 | moderate 0.70-0.85 | breaking < 0.70"
    echo ""
    echo -e "${BOLD}Compliance Pipeline (mandatory):${NC}"
    echo "  1. CFAA  2. ToS  3. robots.txt  4. Rate Limit  5. Fetch  6. Anti-Bot  7. GDPR/CCPA"
    echo ""
    echo -e "${BOLD}Agent Skills (./skills.sh skills):${NC}"
    if [ -d "$SKILLS_DIR" ]; then
        for skill_dir in "${SKILLS_DIR}"/*/; do
            [ -d "$skill_dir" ] || continue
            [ -f "${skill_dir}SKILL.md" ] || continue
            local sname
            sname=$(_extract_frontmatter_field "${skill_dir}SKILL.md" "name")
            printf "  %-28s %s\n" "$sname" "${skill_dir}SKILL.md"
        done
    fi
    echo ""
    echo -e "${BOLD}Package Layout:${NC}"
    echo "  fingerprint/"
    echo "    core/        Orchestrator, fetcher, verbose logging"
    echo "    adaptive/    DOM analysis, change detection, strategy learning"
    echo "    ml/          Embeddings, Ollama Cloud, page classifier"
    echo "    storage/     Redis: structures, embeddings, review queue"
    echo "    compliance/  robots.txt (RFC 9309), rate limiter, bot detector"
    echo "    legal/       CFAA, ToS, GDPR, CCPA"
    echo "    extraction/  Content extractor, file writer, format handlers"
    echo "    alerting/    Change monitor, review queue, notifiers"
    echo ""
}

# ===========================================================================
# doctor - Verify prerequisites
# ===========================================================================
cmd_doctor() {
    log_step "Checking prerequisites"
    local all_ok=true

    echo -n "  Python 3.11+ ... "
    local py_found=false
    for cmd in python3.11 python3.12 python3.13 python3; do
        if command -v "$cmd" &>/dev/null; then
            local ver major minor
            ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
                echo -e "${GREEN}OK${NC} ($cmd $ver)"
                py_found=true
                break
            fi
        fi
    done
    [ "$py_found" = false ] && { echo -e "${RED}MISSING${NC}"; all_ok=false; }

    echo -n "  pip ... "
    if command -v pip3 &>/dev/null || command -v pip &>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}MISSING${NC}"; all_ok=false
    fi

    echo -n "  Docker ... "
    if command -v docker &>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}NOT FOUND${NC} (optional)"
    fi

    echo -n "  Redis ... "
    if command -v redis-cli &>/dev/null && redis-cli -p "${REDIS_PORT}" ping 2>/dev/null | grep -q PONG; then
        echo -e "${GREEN}OK${NC} (localhost:${REDIS_PORT})"
    elif docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${REDIS_CONTAINER_NAME}$"; then
        echo -e "${GREEN}OK${NC} (Docker)"
    else
        echo -e "${YELLOW}NOT RUNNING${NC} (run: ./skills.sh redis)"
    fi

    echo -n "  Git ... "
    if command -v git &>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}MISSING${NC}"; all_ok=false
    fi

    echo -n "  Virtual env ... "
    if [ -d "$VENV_DIR" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}NOT CREATED${NC} (run: ./skills.sh deps)"
    fi

    echo -n "  Skills ... "
    local skill_count=0
    if [ -d "$SKILLS_DIR" ]; then
        for sd in "${SKILLS_DIR}"/*/; do
            [ -f "${sd}SKILL.md" ] && skill_count=$((skill_count + 1))
        done
    fi
    if [ "$skill_count" -gt 0 ]; then
        echo -e "${GREEN}${skill_count} skills found${NC}"
    else
        echo -e "${YELLOW}NONE${NC}"
    fi

    echo ""
    if [ "$all_ok" = true ]; then
        log_ok "All required prerequisites available"
    else
        log_error "Some prerequisites missing"
        exit 1
    fi
}

# ===========================================================================
# help
# ===========================================================================
cmd_help() {
    echo ""
    echo -e "${BOLD}Adaptive Fingerprint - skills.sh${NC}"
    echo ""
    echo "Usage: ./skills.sh <command> [options]"
    echo ""
    echo -e "${BOLD}Agent Skills:${NC}"
    echo "  skills              List all Agent Skills (SKILL.md format)"
    echo "  skills show <name>  Display a skill's full content"
    echo "  skills validate     Validate all skills against the spec"
    echo "  skills tree         Show skills directory tree"
    echo ""
    echo -e "${BOLD}Project Setup:${NC}"
    echo "  scaffold      Create project directories and source file stubs"
    echo "  deps          Create venv and install dependencies"
    echo "  config        Generate config.yaml and .env files"
    echo "  redis         Start Redis via Docker (port ${REDIS_PORT})"
    echo "  redis-stop    Stop the Redis container"
    echo ""
    echo -e "${BOLD}Development:${NC}"
    echo "  lint          Run ruff linter"
    echo "  test          Run pytest suite"
    echo "  clean         Remove __pycache__, .pytest_cache, egg-info"
    echo "  clean --all   Also remove the virtual environment"
    echo ""
    echo -e "${BOLD}Information:${NC}"
    echo "  info          Architecture and module summary"
    echo "  doctor        Check prerequisites"
    echo "  help          Show this help"
    echo ""
    echo "Quick start:"
    echo "  ./skills.sh doctor && ./skills.sh scaffold && ./skills.sh config && ./skills.sh deps"
    echo "  ./skills.sh skills            # See Agent Skills"
    echo "  source .venv/bin/activate"
    echo "  fingerprint analyze --url https://example.com"
    echo ""
}

# ===========================================================================
# Utility functions
# ===========================================================================

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
        skills)     cmd_skills "$@" ;;
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
