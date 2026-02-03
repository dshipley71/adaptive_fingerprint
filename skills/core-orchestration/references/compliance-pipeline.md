# Compliance Pipeline Reference

Every URL fetch passes through this ordered pipeline. Each step must succeed before the next runs.

## Pipeline Steps

### 1. CFAA Authorization Check

```python
result = await cfaa_checker.is_authorized(url)
if not result.authorized:
    raise CFAAViolationError(url, reason=result.reason)
```

Blocks login areas, API endpoints, internal paths, auth query params, non-HTTP schemes.

### 2. Terms of Service Check

```python
result = await tos_checker.check(url)
if not result.allowed:
    raise ToSViolationError(url, directive=result.directive)
```

Pre-fetch check only (response-based checks happen in step 6b).

### 3. robots.txt Check (RFC 9309)

```python
allowed = await robots_checker.is_allowed(url)
if not allowed:
    raise RobotsBlockedError(url)

crawl_delay = await robots_checker.get_crawl_delay(domain)
if crawl_delay:
    rate_limiter.set_minimum_delay(domain, crawl_delay)
```

### 4. Rate Limiter

```python
await rate_limiter.acquire(domain)
```

Blocks until the per-domain delay has elapsed. Respects Crawl-delay from step 3.

### 5. HTTP Fetch

```python
response = await http_client.get(url, headers=headers, timeout=config.timeout)
```

Uses `httpx.AsyncClient` with configured user agent and timeout.

### 6. Post-Fetch Checks

#### 6a. Anti-Bot Detection

```python
bot_result = await bot_detector.check(response)
if bot_result.is_blocked:
    if bot_result.retry_after:
        rate_limiter.set_backoff(domain, bot_result.retry_after)
    raise BotDetectedError(url, block_type=bot_result.block_type)
```

#### 6b. ToS Response Check

```python
tos_result = await tos_checker.check(url, response)
if not tos_result.allowed:
    raise ToSViolationError(url, directive=tos_result.directive)
```

Checks X-Robots-Tag headers and meta robots tags in response.

#### 6c. GDPR Processing

```python
response = await gdpr_handler.process(response)
```

Scans for PII, applies redaction/pseudonymization/skip based on config.

#### 6d. CCPA Processing

```python
response = await ccpa_handler.process(response)
```

Checks for GPC signals and "Do Not Sell" indicators.

### 7. Return Processed Response

Return the compliance-processed response to the caller.

## Error Handling

| Exception | Step | Recovery |
|-----------|------|----------|
| `CFAAViolationError` | 1 | Skip URL permanently |
| `ToSViolationError` | 2, 6b | Skip URL permanently |
| `RobotsBlockedError` | 3 | Skip URL, retry after robots.txt cache expires |
| `RateLimitExceededError` | 4 | Wait and retry |
| `HTTPTimeoutError` | 5 | Retry with backoff |
| `BotDetectedError` | 6a | Stop fetching domain |
| `CaptchaEncounteredError` | 6a | Stop fetching domain |
| `GDPRViolationError` | 6c | Skip content (when mode=skip) |

## Logging

Each step logs its entry and result at the appropriate verbose level. Failed checks log at WARN level with the reason for blocking.
