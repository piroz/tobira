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
  "language": "ja"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Email text (max 100 KB) |
| `headers` | object | No | Email headers for header-based scoring |
| `language` | string | No | ISO 639-1 language code |

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
  }
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

### `GET /health`

Check server health status.

**Response**

```json
{
  "status": "ok"
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
| `source` | string | Yes | Feedback source identifier |

**Response**

```json
{
  "status": "ok",
  "id": "fb-20240101-abc123"
}
```

### `GET /dashboard`

Web dashboard for monitoring predictions and performance. Available when the dashboard is enabled in configuration.

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
