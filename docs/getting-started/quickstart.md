# Quick Start

Get tobira running in 5 minutes.

## Prerequisites

- Python 3.9+
- A trained spam classification model (or use one from HuggingFace Hub)

## 1. Install tobira

```bash
pip install tobira[serving,fasttext]
```

## 2. Start the API server

```bash
tobira serve --backend fasttext --model-path /path/to/model.bin
```

Or with a TOML configuration file:

```bash
tobira serve --config tobira.toml
```

Example `tobira.toml`:

```toml
[backend]
type = "fasttext"
model_path = "/var/lib/tobira/fasttext-spam.bin"
```

The server starts on `http://127.0.0.1:8000` by default.

## 3. Test the API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Buy cheap watches now! Limited offer!"}'
```

Response:

```json
{
  "label": "spam",
  "score": 0.95,
  "labels": {"spam": 0.95, "ham": 0.05}
}
```

## 4. Check health

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

## 5. Set up your MTA

Choose your MTA integration guide:

- [rspamd](../mta/rspamd.md)
- [SpamAssassin](../mta/spamassassin.md)
- [Haraka](../mta/haraka.md)
- [Postfix milter](../mta/postfix-milter.md)

## Next Steps

- [Configuration](configuration.md) — Detailed configuration options
- [Backends](backends.md) — Choose the right backend for your needs
- [Deployment Guide](phased-rollout.md) — Production rollout strategy
