---
name: adaptive-fingerprinting
description: Implement rules-based DOM structure analysis, change detection, and CSS selector inference for the Adaptive Fingerprint system. Use when building or modifying fingerprint/adaptive/ modules including DOMStructureAnalyzer, ChangeDetector, and StrategyLearner. Covers DOM hierarchy parsing, tag frequency analysis, CSS class mapping, semantic landmark detection, change classification (cosmetic/minor/moderate/breaking), and content extraction strategy learning.
---

# Adaptive Fingerprinting

Implement `fingerprint/adaptive/structure_analyzer.py`, `fingerprint/adaptive/change_detector.py`, and `fingerprint/adaptive/strategy_learner.py`.

## DOMStructureAnalyzer

Parses raw HTML into a `PageStructure` fingerprint. Uses BeautifulSoup with the `lxml` parser.

```python
class DOMStructureAnalyzer:
    def analyze(self, html: str, domain: str) -> PageStructure: ...
```

### Analysis Pipeline

1. Parse HTML with `BeautifulSoup(html, "lxml")`.
2. Build `TagHierarchy`: tag counts, depth distribution, parent-child pairs, max depth.
3. Build `css_class_map`: `dict[str, int]` mapping each CSS class to its occurrence count.
4. Collect `id_attributes`: `set[str]` of all `id` values.
5. Detect `semantic_landmarks` by scanning for `<header>`, `<nav>`, `<main>`, `<footer>`, `<article>`, `<aside>`, `<section>`, and ARIA `role` attributes. Map landmark name to its CSS selector.
6. Identify `content_regions` with primary + fallback CSS selectors and confidence scores.
7. Analyze `script_signatures`: collect `src` attributes from `<script>` tags.
8. Detect `detected_framework` from script patterns (React, Vue, Angular, Next.js, etc.).
9. Compute `content_hash` as SHA-256 of the body text content.

### Framework Detection Patterns

| Pattern | Framework |
|---------|-----------|
| `react` in scripts or `data-reactroot` | React |
| `vue` in scripts or `data-v-` attributes | Vue |
| `angular` in scripts or `ng-` attributes | Angular |
| `__next` or `_next` in scripts | Next.js |
| `nuxt` in scripts | Nuxt |
| `gatsby` in scripts | Gatsby |

### Page Type Inference

Infer `page_type` from landmark analysis and tag distribution:

| Indicators | Page Type |
|------------|-----------|
| `<article>` present, high `<p>` count | `article` |
| Many `<li>` items with links | `listing` |
| `<form>`, price patterns, product schema | `product` |
| Hero section, multiple `<section>` blocks | `home` |

## ChangeDetector

Compare two `PageStructure` objects and produce a `ChangeAnalysis`.

```python
class ChangeDetector:
    def detect_changes(self, old: PageStructure, new: PageStructure) -> ChangeAnalysis: ...
    def classify_similarity(self, similarity: float) -> ChangeClassification: ...
```

### Similarity Calculation

Compute an overall weighted similarity from these components:

| Component | Weight | Method |
|-----------|--------|--------|
| Tag similarity | 0.30 | Jaccard coefficient of tag-count keys |
| Class similarity | 0.25 | Jaccard coefficient of CSS class sets |
| Landmark similarity | 0.20 | Exact match ratio of landmark keys |
| Depth similarity | 0.15 | Normalized difference of max depths |
| Script similarity | 0.10 | Jaccard coefficient of script signature sets |

### Change Classification Thresholds

| Classification | Similarity Range |
|----------------|-----------------|
| `COSMETIC` | > 0.95 |
| `MINOR` | 0.85 - 0.95 |
| `MODERATE` | 0.70 - 0.85 |
| `BREAKING` | < 0.70 |

### Detecting Specific Changes

Populate `ChangeAnalysis.changes` with `StructureChange` objects by comparing:

- Added/removed tags (`TAG_ADDED`, `TAG_REMOVED`)
- Added/removed/renamed CSS classes (`CLASS_ADDED`, `CLASS_REMOVED`, `CLASS_RENAMED`)
- Changed landmarks (`LANDMARK_CHANGED`)
- Framework changes (`FRAMEWORK_CHANGED`)
- Script changes (`SCRIPT_ADDED`, `SCRIPT_REMOVED`)

### Class Rename Detection

Detect rename patterns by looking for class pairs where:
- One class was removed and another was added
- The Levenshtein distance between names is small relative to length
- Examples: `btn-primary` -> `btn-primary-v2`, `header-nav` -> `header-navigation`

## StrategyLearner

Infer CSS selectors for content extraction from a `PageStructure`.

```python
class StrategyLearner:
    async def learn_strategy(self, structure: PageStructure) -> ExtractionStrategy: ...
```

### Selector Inference Rules

For each content region, generate a primary selector and fallback selectors:

1. Prefer semantic selectors: `article`, `main`, `[role="main"]`
2. Fall back to ID-based: `#content`, `#main-content`
3. Fall back to class-based: `.article-content`, `.post-body`
4. Generate `confidence` based on specificity and uniqueness

### ExtractionStrategy Structure

```python
ExtractionStrategy(
    domain="example.com",
    page_type="article",
    title=SelectorRule(primary="h1", fallbacks=["article h1", ".title"]),
    content=SelectorRule(primary="article", fallbacks=["main", ".content"]),
    metadata={"author": SelectorRule(...), "date": SelectorRule(...)},
)
```

## Verbose Logging

| Operation | Description |
|-----------|-------------|
| `STRUCTURE:ANALYZE` | Starting DOM analysis |
| `STRUCTURE:TAGS` | Tag hierarchy built |
| `STRUCTURE:CLASSES` | CSS class map built |
| `STRUCTURE:LANDMARKS` | Semantic landmarks detected |
| `STRUCTURE:FRAMEWORK` | Framework detected |
| `CHANGE:START` | Starting change detection |
| `CHANGE:SIMILARITY` | Component similarities calculated |
| `CHANGE:CLASSIFY` | Change classified |
| `CHANGE:DETAIL` | Individual change detected |
