---
name: redis-storage
description: Implement the Redis persistence layer for the Adaptive Fingerprint system. Use when building or modifying fingerprint/storage/ modules including StructureStore, EmbeddingStore, ReviewStore, and Cache. Covers structure versioning (up to 10 versions), embedding persistence, manual review queue with sorted sets, volatile domain tracking, TTL-based expiration, and JSON serialization of dataclass models to Redis.
---

# Redis Storage

Implement `fingerprint/storage/structure_store.py`, `fingerprint/storage/embedding_store.py`, `fingerprint/storage/review_store.py`, and `fingerprint/storage/cache.py`.

## Redis Key Schema

All keys use `{prefix}:` namespace (default: `fingerprint`).

```
{prefix}:structure:{domain}:{page_type}:{variant_id}           Current structure
{prefix}:structure:{domain}:{page_type}:{variant_id}:v{n}      Version history (1..max_versions)
{prefix}:embedding:{domain}:{page_type}:{variant_id}           Embedding vector
{prefix}:volatile:{domain}                                      Volatile domain flag
{prefix}:review:pending                                         Sorted set (score=timestamp)
{prefix}:review:item:{id}                                       Individual review item hash
{prefix}:review:domain:{domain}                                 Set of review IDs per domain
{prefix}:review:completed:{id}                                  Archived completed review
```

## StructureStore

```python
class StructureStore:
    def __init__(self, config: RedisConfig): ...
    async def save(self, structure: PageStructure) -> None: ...
    async def get(self, domain: str, page_type: str, variant_id: str = "default") -> PageStructure | None: ...
    async def delete(self, domain: str, page_type: str, variant_id: str = "default") -> bool: ...
    async def list_versions(self, domain: str, page_type: str) -> list[PageStructure]: ...
    async def is_volatile(self, domain: str) -> bool: ...
    async def mark_volatile(self, domain: str, ttl: int = 86400) -> None: ...
```

### Version Management

On each `save()`:

1. Read the current structure at the primary key.
2. If it exists, push it to `v{n+1}` (rotate versions).
3. Write the new structure to the primary key.
4. Trim history to `max_versions` (default 10) by deleting the oldest.
5. Set TTL on all keys (`ttl_seconds`, default 604800 = 7 days).

### Serialization

Convert dataclasses to JSON via `dataclasses.asdict()`. Handle special types:

| Type | Serialization |
|------|---------------|
| `datetime` | ISO 8601 string |
| `set` | Sorted list |
| `Enum` | `.value` |
| `None` | Omit from dict |

Deserialize back by reconstructing `TagHierarchy`, `ContentRegion`, etc.

## EmbeddingStore

```python
class EmbeddingStore:
    async def save(self, embedding: StructureEmbedding) -> None: ...
    async def get(self, domain: str, page_type: str, variant_id: str = "default") -> StructureEmbedding | None: ...
    async def delete(self, domain: str, page_type: str, variant_id: str = "default") -> bool: ...
```

Store the full `StructureEmbedding` including `vector` (list of floats), `dimensions`, `model_name`, and `description`.

## ReviewStore

```python
class ReviewStore:
    async def add(self, item: ReviewItem) -> str: ...
    async def get(self, review_id: str) -> ReviewItem | None: ...
    async def get_pending(self, limit: int = 50, offset: int = 0) -> list[ReviewItem]: ...
    async def approve(self, review_id: str, reviewer: str, notes: str = "") -> None: ...
    async def reject(self, review_id: str, reviewer: str, notes: str = "") -> None: ...
    async def stats(self) -> dict[str, int]: ...
```

### ReviewItem Flow

1. `add()` generates a UUID, stores item hash at `review:item:{id}`, adds ID to `review:pending` sorted set (score = timestamp), adds to `review:domain:{domain}` set.
2. `get_pending()` uses `ZRANGEBYSCORE` with offset/limit for pagination.
3. `approve()`/`reject()` removes from pending, moves to `review:completed:{id}`, updates status.
4. `stats()` returns counts of pending, completed, rejected.

## Cache

Generic in-memory TTL cache for expensive operations.

```python
class Cache[T]:
    def __init__(self, default_ttl: int = 300): ...
    def set(self, key: str, value: T, ttl: int | None = None) -> None: ...
    def get(self, key: str) -> T | None: ...
    def delete(self, key: str) -> bool: ...
    def clear(self) -> None: ...
    def cleanup_expired(self) -> int: ...
    def stats(self) -> dict[str, int]: ...
```

Store entries as `(value, expiry_timestamp)` tuples. `get()` checks expiry and returns `None` for expired entries.

## Configuration

```yaml
redis:
  url: "redis://localhost:6379/0"
  key_prefix: "fingerprint"
  ttl_seconds: 604800    # 7 days
  max_versions: 10
```

Environment variable override: `REDIS_URL`.

## Verbose Logging

| Operation | Description |
|-----------|-------------|
| `STORE:SAVE` | Structure saved |
| `STORE:GET` | Structure retrieved |
| `STORE:VERSION` | Version rotated |
| `STORE:VOLATILE` | Domain marked volatile |
| `EMBED_STORE:SAVE` | Embedding saved |
| `EMBED_STORE:GET` | Embedding retrieved |
| `REVIEW:ADD` | Review item added |
| `REVIEW:APPROVE` | Review approved |
| `REVIEW:REJECT` | Review rejected |
| `CACHE:HIT` | Cache hit |
| `CACHE:MISS` | Cache miss |
| `CACHE:EXPIRE` | Expired entries cleaned |
