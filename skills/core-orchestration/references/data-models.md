# Data Models Reference

All models defined in `fingerprint/models.py` using Python dataclasses.

## Enums

### FingerprintMode

```python
class FingerprintMode(Enum):
    RULES = "rules"
    ML = "ml"
    ADAPTIVE = "adaptive"
```

### ChangeClassification

```python
class ChangeClassification(Enum):
    COSMETIC = "cosmetic"    # > 0.95 similarity
    MINOR = "minor"          # 0.85 - 0.95
    MODERATE = "moderate"    # 0.70 - 0.85
    BREAKING = "breaking"    # < 0.70
```

### ChangeType

```python
class ChangeType(Enum):
    TAG_ADDED = "tag_added"
    TAG_REMOVED = "tag_removed"
    CLASS_RENAMED = "class_renamed"
    CLASS_ADDED = "class_added"
    CLASS_REMOVED = "class_removed"
    ID_CHANGED = "id_changed"
    STRUCTURE_REORGANIZED = "structure_reorganized"
    CONTENT_RELOCATED = "content_relocated"
    LANDMARK_CHANGED = "landmark_changed"
    SCRIPT_ADDED = "script_added"
    SCRIPT_REMOVED = "script_removed"
    FRAMEWORK_CHANGED = "framework_changed"
    NAVIGATION_CHANGED = "navigation_changed"
    PAGINATION_CHANGED = "pagination_changed"
    MINOR_LAYOUT_SHIFT = "minor_layout_shift"
    MAJOR_REDESIGN = "major_redesign"
```

### ReviewStatus

```python
class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
```

### AlertSeverity

```python
class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
```

## Core Dataclasses

### PageStructure

Primary fingerprint model. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `domain` | `str` | Domain name |
| `page_type` | `str` | Classified page type |
| `url_pattern` | `str` | URL pattern |
| `variant_id` | `str` | Structure variant (default: "default") |
| `tag_hierarchy` | `TagHierarchy \| None` | Tag structure analysis |
| `css_class_map` | `dict[str, int]` | CSS class -> occurrence count |
| `id_attributes` | `set[str]` | HTML id values |
| `semantic_landmarks` | `dict[str, str]` | Landmark name -> CSS selector |
| `content_regions` | `list[ContentRegion]` | Identified extraction zones |
| `navigation_selectors` | `list[str]` | Navigation element selectors |
| `script_signatures` | `list[str]` | Script src attributes |
| `detected_framework` | `str \| None` | Detected JS framework |
| `captured_at` | `datetime` | Capture timestamp |
| `version` | `int` | Structure version |
| `content_hash` | `str` | SHA-256 of body text |
| `description` | `str` | LLM-generated description |

### ChangeAnalysis

Result from one fingerprinting mode comparison.

| Field | Type | Description |
|-------|------|-------------|
| `similarity` | `float` | 0.0-1.0 similarity score |
| `mode_used` | `FingerprintMode` | Which mode produced this |
| `classification` | `ChangeClassification` | Severity level |
| `breaking` | `bool` | Whether change is breaking |
| `changes` | `list[StructureChange]` | Individual detected changes |
| `fields_affected` | `dict[str, str]` | Affected field -> description |
| `can_auto_adapt` | `bool` | Whether auto-adaptation is possible |
| `adaptation_confidence` | `float` | Confidence in auto-adaptation |
| `reason` | `str` | Human-readable summary |
| `escalated` | `bool` | Whether adaptive mode escalated |
| `escalation_triggers` | `list[EscalationTrigger]` | What triggered escalation |
| `duration_ms` | `float` | Processing time |

### StructureEmbedding

| Field | Type | Description |
|-------|------|-------------|
| `domain` | `str` | Domain name |
| `page_type` | `str` | Page type |
| `variant_id` | `str` | Variant identifier |
| `vector` | `list[float]` | Embedding vector |
| `dimensions` | `int` | Vector dimensions (384) |
| `model_name` | `str` | Model used |
| `description` | `str` | Text description that was embedded |
| `generated_at` | `datetime` | Generation timestamp |

### ExtractionStrategy

| Field | Type | Description |
|-------|------|-------------|
| `domain` | `str` | Target domain |
| `page_type` | `str` | Page type |
| `version` | `int` | Strategy version |
| `title` | `SelectorRule \| None` | Title extraction rule |
| `content` | `SelectorRule \| None` | Content extraction rule |
| `metadata` | `dict[str, SelectorRule]` | Named metadata rules |
| `learned_at` | `datetime` | When strategy was learned |
| `learning_source` | `str` | "initial", "adaptation", "manual" |
| `confidence_scores` | `dict[str, float]` | Per-field confidence |

### ReviewItem

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | UUID |
| `domain` | `str` | Domain |
| `page_type` | `str` | Page type |
| `change_analysis` | `ChangeAnalysis` | The detected change |
| `status` | `ReviewStatus` | Current status |
| `created_at` | `datetime` | Queue entry time |
| `reviewed_at` | `datetime \| None` | Review time |
| `reviewer` | `str` | Reviewer identifier |
| `notes` | `str` | Review notes |
