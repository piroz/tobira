# API Reference

The tobira API server exposes a REST API for spam classification, health checks, and feedback collection.

## Base URL

```
http://127.0.0.1:8000
```

## Endpoints

### `POST /predict`

Classify email text as spam or ham.

**Request**

```json
{
  "text": "Email subject and body text",
  "headers": {
    "spf": "pass",
    "dkim": "pass",
    "dmarc": "pass",
    "from_addr": "sender@example.com",
    "reply_to": "reply@example.com",
    "received": ["from mail.example.com ..."],
    "content_type": "text/plain"
  },
  "language": "ja",
  "explain": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Email text (max 100 KB) |
| `headers` | object | No | Email headers for header-based scoring |
| `language` | string | No | ISO 639-1 language code |
| `explain` | boolean | No | Return token-level attribution explanations |

**Response**

```json
{
  "label": "spam",
  "score": 0.95,
  "labels": {
    "spam": 0.95,
    "ham": 0.05
  },
  "header_score": 0.8,
  "language": "en",
  "ai_generated": {
    "detected": false,
    "confidence": 0.12,
    "indicators": []
  },
  "explanations": [
    {"token": "Congratulations", "score": 0.42},
    {"token": "won", "score": 0.31}
  ],
  "model_version": "fasttext-v1"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Best classification label |
| `score` | float | Spam probability (0.0 to 1.0) |
| `labels` | object | Scores for all classes |
| `header_score` | float | Header-based risk score (if enabled) |
| `language` | string | Detected or provided language |
| `ai_generated` | object | AI-generated text detection (if enabled) |
| `explanations` | array | Token-level attribution (when `explain: true`) |
| `model_version` | string | Model variant used (visible during A/B testing) |

### `GET /health`

Check server health status.

**Response**

```json
{
  "status": "ok"
}
```

### `GET /health/ready`

Kubernetes readiness probe. Returns `200` when the server is ready to accept traffic.

**Response**

```json
{
  "ready": true
}
```

When not ready:

```json
{
  "ready": false,
  "reason": "model loading"
}
```

### `GET /health/live`

Kubernetes liveness probe. Returns `200` when the server process is alive.

**Response**

```json
{
  "alive": true
}
```

### `POST /feedback`

Submit feedback on a classification result. Available when feedback collection is enabled.

**Request**

```json
{
  "text": "The email text that was misclassified",
  "label": "ham",
  "source": "user-report"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Email text |
| `label` | string | Yes | Correct label (`spam` or `ham`) |
| `source` | string | No | Feedback source identifier (default: `"unknown"`) |

**Response**

```json
{
  "status": "ok",
  "id": "fb-20240101-abc123"
}
```

### `GET /dashboard`

Web dashboard for monitoring predictions and performance. Available when the dashboard is enabled in configuration.

## Active Learning Endpoints

Available when active learning is enabled in configuration. Items are automatically added to the queue during prediction when the model's uncertainty exceeds a configured threshold.

### `GET /v1/active-learning/queue`

Retrieve queued items sorted by uncertainty score.

**Response**

```json
{
  "samples": [
    {
      "id": "abc-123",
      "text": "Check this limited offer...",
      "score": 0.55,
      "labels": {"spam": 0.55, "ham": 0.45},
      "uncertainty": 0.99,
      "strategy": "entropy",
      "timestamp": "2025-01-01T00:00:00+00:00",
      "labeled": false,
      "assigned_label": null
    }
  ],
  "total": 1,
  "pending": 1,
  "labeled": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `samples` | array | Queued samples sorted by uncertainty |
| `total` | integer | Total samples in queue |
| `pending` | integer | Unlabeled samples |
| `labeled` | integer | Labeled samples |

### `POST /v1/active-learning/label`

Submit a label for an active learning sample.

**Request**

```json
{
  "sample_id": "abc-123",
  "label": "spam"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sample_id` | string | Yes | ID of the sample to label |
| `label` | string | Yes | Label value (`spam` or `ham`) |

**Response**

```json
{
  "status": "labeled",
  "sample_id": "abc-123",
  "label": "spam"
}
```

### `GET /v1/active-learning/stats`

Retrieve statistics about the active learning queue.

**Response**

```json
{
  "total": 42,
  "pending": 15,
  "labeled": 27,
  "label_counts": {"spam": 20, "ham": 7}
}
```

## A/B Testing

Available when A/B testing is configured. The server automatically routes `/predict` requests to model variants based on the configured strategy (random or hash-based). The `model_version` field in predict responses indicates which variant was used.

### `GET /api/ab-test/results`

Retrieve A/B test results with per-variant metrics.

**Response**

```json
{
  "variants": {
    "fasttext-v1": {
      "predictions": 5200,
      "avg_latency_ms": 2.1,
      "avg_score": 0.82,
      "label_counts": {"spam": 3100, "ham": 2100},
      "errors": 0
    },
    "onnx-v2": {
      "predictions": 4800,
      "avg_latency_ms": 28.5,
      "avg_score": 0.89,
      "label_counts": {"spam": 2900, "ham": 1900},
      "errors": 3
    }
  }
}
```

## Error Responses

All endpoints return standard HTTP error codes:

| Code | Description |
|------|-------------|
| `400` | Invalid request (missing or malformed fields) |
| `422` | Validation error (text too long, invalid language code) |
| `500` | Internal server error (backend failure) |

Error response format:

```json
{
  "detail": "Error description"
}
```

## Email Headers Object

The optional `headers` field in `/predict` requests allows header-based risk scoring. This blends with the text-based score for improved accuracy.

| Field | Type | Description |
|-------|------|-------------|
| `spf` | string | SPF result (`pass`, `fail`, `softfail`, `none`) |
| `dkim` | string | DKIM result (`pass`, `fail`, `none`) |
| `dmarc` | string | DMARC result (`pass`, `fail`, `none`) |
| `from_addr` | string | From header address |
| `reply_to` | string | Reply-To header address |
| `received` | string[] | List of Received headers |
| `content_type` | string | Content-Type header value |
