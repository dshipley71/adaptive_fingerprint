---
name: legal-compliance
description: Implement legal compliance checking for the Adaptive Fingerprint system. Use when building or modifying fingerprint/legal/ modules including CFAAChecker, ToSChecker, GDPRHandler, and CCPAHandler. Covers CFAA authorization enforcement (blocking login areas, API endpoints, internal paths), Terms of Service respect (meta robots, X-Robots-Tag), GDPR PII detection and handling (redact/pseudonymize/skip for email, phone, IP, SSN, credit card), and CCPA Global Privacy Control signal compliance.
---

# Legal Compliance

Implement `fingerprint/legal/cfaa_checker.py`, `fingerprint/legal/tos_checker.py`, `fingerprint/legal/gdpr_handler.py`, and `fingerprint/legal/ccpa_handler.py`.

## CFAAChecker

Enforce Computer Fraud and Abuse Act compliance by only accessing publicly authorized content.

```python
@dataclass
class AuthorizationResult:
    authorized: bool
    reason: str = ""
    risk_level: str = "none"  # none, low, medium, high

class CFAAChecker:
    async def is_authorized(self, url: str) -> AuthorizationResult: ...
    def check_response_headers(self, headers: dict[str, str]) -> AuthorizationResult: ...
```

### URL Path Blocking

Block these path patterns (case-insensitive regex match):

**Authentication paths** (risk: high):
```
/login, /signin, /sign-in, /auth, /oauth, /sso, /account,
/my-account, /dashboard, /admin, /user/, /profile, /settings,
/preferences, /private, /members, /secure
```

**API endpoints** (risk: medium):
```
^/api/, ^/v\d+/, ^/graphql, ^/rest/, ^/_api/, /api$, \.json$, \.xml$
```

**Internal/system paths** (risk: high):
```
^/\., ^/_, /\.git, /\.env, /wp-admin, /phpmyadmin,
/cpanel, /cgi-bin, /server-status, /config
```

### Query Parameter Blocking

Block URLs containing auth-related query parameters (risk: medium):
`token=`, `key=`, `apikey=`, `api_key=`, `auth=`, `session=`

### Scheme Enforcement

Only allow `http` and `https` schemes.

### Response Header Check

Block if `WWW-Authenticate` header is present (server requires authentication).

## ToSChecker

Respect website Terms of Service directives.

```python
@dataclass
class ToSResult:
    allowed: bool
    directive: str = ""
    details: dict[str, Any] | None = None

class ToSChecker:
    async def check(self, url: str, response: Any | None = None) -> ToSResult: ...
```

### X-Robots-Tag Header

Check the `X-Robots-Tag` response header for directives:

| Directive | Action |
|-----------|--------|
| `noindex` | Block (if `respect_noindex` enabled) |
| `nofollow` | Block (if `respect_nofollow` enabled) |
| `noarchive` | Block |
| `none` | Block (equivalent to noindex + nofollow) |

### Meta Robots Tags

Parse HTML for `<meta name="robots" content="...">` and check the same directives as above.

### Link Extraction

`extract_nofollow_links(content)` returns URLs from `<a rel="nofollow">` tags for informational use.

## GDPRHandler

Detect and handle personally identifiable information.

```python
@dataclass
class PIIMatch:
    pii_type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0

@dataclass
class PIIDetectionResult:
    contains_pii: bool
    matches: list[PIIMatch]
    pii_types_found: set[str]

class GDPRHandler:
    async def process(self, response: Any) -> Any: ...
    def scan(self, content: str) -> PIIDetectionResult: ...
    def handle(self, content: str, result: PIIDetectionResult) -> str: ...
```

### PII Detection Patterns

| PII Type | Regex Pattern |
|----------|---------------|
| `email` | `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z\|a-z]{2,}\b` |
| `phone` | US format: `\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b` |
| `ip_address` | IPv4 dotted notation with valid octets |
| `ssn` | `\b(?!000\|666\|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b` |
| `credit_card` | Visa, Mastercard, Amex, Discover patterns |
| `postal_code` | US ZIP: `\b[0-9]{5}(?:-[0-9]{4})?\b` |
| `phone_eu` | EU/UK formats: `\b\+?(?:44\|33\|49\|39\|34\|31\|32\|43\|41)[0-9\s-]{8,12}\b` |

### Handling Modes

| Mode | Behavior |
|------|----------|
| `redact` | Replace each match with `[REDACTED-{TYPE}]` |
| `pseudonymize` | Replace with consistent hash-based pseudonym `[{TYPE}-{sha256[:8]}]` |
| `skip` | Raise `GDPRViolationError`, do not process the content |

When redacting or pseudonymizing, process matches in reverse position order to maintain correct offsets.

## CCPAHandler

Respect California Consumer Privacy Act signals.

```python
@dataclass
class CCPACheckResult:
    compliant: bool
    opt_out_detected: bool = False
    gpc_enabled: bool = False
    reason: str = ""

class CCPAHandler:
    def check_request_headers(self, headers: dict[str, str]) -> CCPACheckResult: ...
    async def process(self, response: Any) -> Any: ...
```

### Global Privacy Control

Check for `Sec-GPC: 1` header. When detected, log and flag the opt-out.

### "Do Not Sell" Detection

Scan response body (case-insensitive) for:
```
do not sell, do not sell my personal information, do not sell my info,
opt-out of sale, opt out of sale, ccpa opt-out, california privacy
```

## Configuration

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
    pii_handling: "redact"
    log_pii_access: true
  ccpa:
    enabled: true
    respect_opt_out: true
    respect_gpc: true
```

## Verbose Logging

| Operation | Description |
|-----------|-------------|
| `CFAA:CHECK` | Checking URL authorization |
| `CFAA:AUTHORIZED` | Access authorized |
| `CFAA:BLOCKED` | Access blocked (reason, risk_level) |
| `TOS:CHECK` | Checking ToS directives |
| `TOS:NOINDEX` | noindex directive found |
| `TOS:NOFOLLOW` | nofollow directive found |
| `GDPR:SCAN` | Scanning content for PII |
| `GDPR:PII_FOUND` | PII detected (types, count) |
| `GDPR:REDACT` | Redacting PII |
| `GDPR:PSEUDONYMIZE` | Pseudonymizing PII |
| `CCPA:GPC` | GPC signal detected |
| `CCPA:OPT_OUT` | "Do Not Sell" indicator found |

## References

- [agents-spec.md](references/agents-spec.md) - Complete Python implementation with full source code, PII regex patterns, and path blocklists
