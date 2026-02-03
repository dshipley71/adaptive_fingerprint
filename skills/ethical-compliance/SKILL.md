---
name: ethical-compliance
description: Implement ethical web access compliance for the Adaptive Fingerprint system. Use when building or modifying fingerprint/compliance/ modules including RobotsParser (RFC 9309), RateLimiter, and BotDetector. Covers robots.txt parsing with Crawl-delay support, adaptive per-domain rate limiting with exponential backoff, and anti-bot detection that respects CAPTCHAs, block pages, and JS challenges rather than evading them.
---

# Ethical Compliance

Implement `fingerprint/compliance/robots_parser.py`, `fingerprint/compliance/rate_limiter.py`, and `fingerprint/compliance/bot_detector.py`.

## RobotsParser (RFC 9309)

Full implementation of [RFC 9309](https://www.rfc-editor.org/rfc/rfc9309.html).

```python
class RobotsParser:
    def parse(self, content: str, user_agent: str) -> RobotsData: ...

class RobotsChecker:
    async def is_allowed(self, url: str) -> bool: ...
    async def get_crawl_delay(self, domain: str) -> float | None: ...
    async def get_sitemaps(self, domain: str) -> list[str]: ...
```

### Parsing Rules

1. Match rules by `User-agent` field (case-insensitive). Match the most specific agent first, fall back to `*`.
2. `Allow` takes precedence over `Disallow` when both match the same path length. Longer path match wins.
3. Support wildcards: `*` matches any sequence, `$` anchors to end of URL.
4. Parse `Crawl-delay` directive (not in RFC 9309 core but widely used).
5. Collect `Sitemap` directives.

### Caching

Cache parsed `RobotsData` per domain with configurable TTL (default 3600s). Fetch `{scheme}://{domain}/robots.txt` on first access.

### Edge Cases

- 404 response: treat as fully allowed.
- 5xx response: treat as fully disallowed (conservative).
- Empty file: treat as fully allowed.
- Encoding: handle UTF-8 BOM.

## RateLimiter

Per-domain adaptive rate limiting.

```python
class RateLimiter:
    async def acquire(self, domain: str) -> None: ...
    async def report_error(self, domain: str, status_code: int) -> None: ...
    async def report_success(self, domain: str, response_time: float) -> None: ...

@dataclass
class DomainState:
    last_request: float
    current_delay: float
    consecutive_errors: int
    backoff_until: float | None
```

### Delay Calculation

1. Start with `default_delay` (1.0s).
2. If robots.txt provides `Crawl-delay`, use the larger of the two.
3. On 429 or 503 response: multiply delay by `backoff_multiplier` (2.0), up to `max_delay` (30.0s).
4. If `Retry-After` header present, respect it.
5. On success: if `adapt_to_response_time` is true and server responds slowly (>2s), increase delay proportionally.
6. Gradually reduce delay back to default after consecutive successes.

### acquire() Behavior

1. Look up `DomainState` for the domain.
2. If `backoff_until` is set and not yet reached, `await anyio.sleep()` until that time.
3. Calculate time since `last_request`.
4. If less than `current_delay`, sleep the remaining time.
5. Update `last_request` to now.

## BotDetector

Detect anti-bot measures and stop gracefully. Never attempt to bypass.

```python
class BotDetector:
    async def check(self, response: Any) -> BotCheckResult: ...

@dataclass
class BotCheckResult:
    is_blocked: bool
    block_type: str          # "none", "captcha", "block_page", "rate_limit", "js_challenge"
    retry_after: float | None
    evidence: str
```

### Detection Patterns

**CAPTCHA detection** - scan response body for:
- reCAPTCHA: `recaptcha`, `g-recaptcha`
- hCaptcha: `hcaptcha`, `h-captcha`
- Cloudflare Turnstile: `cf-turnstile`
- Generic: `captcha` in form elements

**Block page detection** - look for:
- HTTP 403 with specific body patterns: `access denied`, `blocked`, `forbidden`
- Cloudflare challenge: `cf-browser-verification`, `_cf_chl`
- AWS WAF: `awswaf`

**Rate limit detection**:
- HTTP 429 status
- `Retry-After` header (parse as seconds or HTTP date)

**JS challenge detection**:
- Response requires JavaScript execution to proceed
- `<noscript>` blocks with challenge content
- Cloudflare `__cf_chl_jschl_tk__`

### Response Behavior

When `is_blocked=True`, the compliance pipeline raises `BotDetectedError` (or `CaptchaEncounteredError`). The system stops fetching that domain.

## Configuration

```yaml
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
```

## Verbose Logging

| Operation | Description |
|-----------|-------------|
| `ROBOTS:FETCH` | Fetching robots.txt for domain |
| `ROBOTS:PARSE` | Parsing robots.txt content |
| `ROBOTS:ALLOWED` | URL is allowed |
| `ROBOTS:BLOCKED` | URL is blocked by robots.txt |
| `ROBOTS:CRAWL_DELAY` | Crawl-delay directive found |
| `RATELIMIT:ACQUIRE` | Acquiring rate limit slot |
| `RATELIMIT:WAIT` | Waiting for rate limit delay |
| `RATELIMIT:BACKOFF` | Backing off after error |
| `RATELIMIT:ADAPT` | Adapting delay to response time |
| `ANTIBOT:CHECK` | Checking for anti-bot measures |
| `ANTIBOT:CAPTCHA` | CAPTCHA detected |
| `ANTIBOT:BLOCKED` | Block page detected |
| `ANTIBOT:RATELIMIT` | Rate limit response detected |
