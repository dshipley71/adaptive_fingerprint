# Exception Hierarchy Reference

All exceptions defined in `fingerprint/exceptions.py`. All inherit from `FingerprintError`.

```
FingerprintError
├── AnalysisError
│   ├── InvalidHTMLError
│   └── EmptyContentError
├── ChangeDetectionError
│   └── IncompatibleStructuresError
├── MLError
│   ├── EmbeddingError
│   └── ModelLoadError
├── OllamaCloudError
│   ├── OllamaAuthError
│   ├── OllamaTimeoutError
│   └── OllamaRateLimitError
├── StorageError
│   ├── RedisConnectionError
│   └── SerializationError
├── FetchError
│   ├── HTTPTimeoutError
│   └── HTTPStatusError
├── ComplianceError
│   ├── RobotsBlockedError
│   ├── RateLimitExceededError
│   ├── CrawlDelayError
│   └── BotDetectedError
│       └── CaptchaEncounteredError
└── LegalComplianceError
    ├── CFAAViolationError
    │   └── UnauthorizedAccessError
    ├── ToSViolationError
    ├── GDPRViolationError
    └── CCPAViolationError
```

Each exception carries relevant context (URL, domain, reason, etc.) for logging and error handling.
