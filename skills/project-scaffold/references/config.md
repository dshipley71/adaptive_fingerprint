# Configuration Reference

Complete `config.yaml` options for the Adaptive Fingerprint system.

## Fingerprinting

```yaml
fingerprinting:
  mode: adaptive                          # "rules", "ml", or "adaptive"
  adaptive:
    class_change_threshold: 0.15          # Escalate to ML if >15% classes changed
    rules_uncertainty_threshold: 0.80     # Escalate to ML if rules similarity < 0.80
    cache_ml_results: true                # Cache ML results to avoid recomputation
  thresholds:
    cosmetic: 0.95                        # > 0.95 = cosmetic change
    minor: 0.85                           # 0.85-0.95 = minor change
    moderate: 0.70                        # 0.70-0.85 = moderate change
    breaking: 0.70                        # < 0.70 = breaking change
```

## Ollama Cloud

```yaml
ollama_cloud:
  enabled: true                           # Enable LLM descriptions
  model: "gemma3:12b"                     # Model to use
  timeout: 30                             # Request timeout (seconds)
  max_retries: 3                          # Max retry attempts
  temperature: 0.3                        # LLM temperature
  max_tokens: 500                         # Max response tokens
```

## Embeddings

```yaml
embeddings:
  model: "all-MiniLM-L6-v2"              # Sentence transformer model
  cache_embeddings: true                  # Cache generated embeddings
```

## Redis

```yaml
redis:
  url: "redis://localhost:6379/0"         # Connection URL
  key_prefix: "fingerprint"              # Key namespace prefix
  ttl_seconds: 604800                     # Key expiration (7 days)
  max_versions: 10                        # Max structure versions to keep
```

## HTTP

```yaml
http:
  user_agent: "AdaptiveFingerprint/1.0 (+https://example.com/bot-info)"
  timeout: 30
  max_retries: 3
```

## Compliance

```yaml
compliance:
  robots_txt:
    enabled: true
    cache_ttl: 3600                       # Cache robots.txt for 1 hour
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

## Legal

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
    pii_handling: "redact"                # "redact", "pseudonymize", or "skip"
    log_pii_access: true
  ccpa:
    enabled: true
    respect_opt_out: true
    respect_gpc: true
```

## Extraction

```yaml
extraction:
  enabled: true
  output_dir: "./extracted"
  formats: ["json", "csv"]
  include_metadata: true
  include_html: false
  max_content_length: 1000000
```

## Alerting

```yaml
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
```

## Verbose

```yaml
verbose:
  enabled: true
  level: 2                                # 0=errors, 1=warnings, 2=info, 3=debug
  format: "structured"                    # "structured" or "plain"
  include_timestamp: true
```
