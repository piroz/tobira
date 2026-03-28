# tobira

**ML-powered spam detection toolkit for mail servers.**

Add modern ML-based spam classification to your existing mail infrastructure — SpamAssassin, rspamd, Haraka, or Postfix — without ML expertise.

[![PyPI version](https://img.shields.io/pypi/v/tobira)](https://pypi.org/project/tobira/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

## Features

- **Pluggable backends** — swap inference engines without changing your setup
- **MTA integration** — plugins for rspamd, SpamAssassin, Haraka, and Postfix (milter)
- **CLI toolkit** — train, evaluate, serve, and monitor from a single command
- **Production-ready API** — FastAPI server with health checks, auth, and monitoring
- **Fail-open design** — graceful degradation when the ML service is unavailable

## Backends

| Backend | Description | Use case |
|---------|------------|----------|
| **FastText** | Linear classifier, very fast | High-throughput / low-resource |
| **BERT** | Transformer fine-tuning | Highest accuracy |
| **ONNX** | Quantized BERT inference | Fast + accurate |
| **LLM API** | OpenAI-compatible APIs | Zero training data needed |
| **Ollama** | Local LLM inference | Privacy-first / air-gapped |
| **Two-Stage** | FastText screening → BERT precision | Balanced throughput + accuracy |
| **Ensemble** | Weighted voting across models | Maximum reliability |

Custom backends can be added via the `tobira.backends` entry point.

## Quick Start

```bash
pip install tobira[serving,fasttext]
```

### 1. Initialize config

```bash
tobira init
# Detects your MTA and generates tobira.toml
```

### 2. Start the API server

```bash
tobira serve --config tobira.toml
```

### 3. Classify an email

```python
import httpx

resp = httpx.post("http://127.0.0.1:8000/v1/predict", json={
    "text": "Congratulations! You've won a free iPhone..."
})
print(resp.json())
# {"label": "spam", "score": 0.98, "labels": {"spam": 0.98, "ham": 0.02}}
```

## MTA Integration

tobira provides ready-to-use plugins for major MTAs:

| MTA | Plugin | Config file |
|-----|--------|-------------|
| rspamd | `tobira.lua` | `tobira.conf` |
| SpamAssassin | `Tobira.pm` | `tobira.cf` |
| Haraka | Node.js plugin | `tobira.ini` |
| Postfix | milter (`tobira milter`) | `tobira-milter.conf` |

All plugins support:
- Configurable API endpoint and timeout
- Fail-open mode (skip ML on API errors)
- Tiered spam scoring (high / medium / low confidence)
- Optional header forwarding (SPF / DKIM / DMARC)

## CLI Commands

```
tobira serve          Start the prediction API server
tobira init           Detect MTA and generate config
tobira doctor         Verify config, backends, and connectivity
tobira train          Fine-tune a model on your labeled data
tobira evaluate       Evaluate model accuracy on a test set
tobira monitor        Analyze prediction logs for drift (PSI, KS)
tobira hub-push/pull  Share models via Hugging Face Hub
tobira distill        Knowledge distillation (teacher → student)
tobira demo           Launch a Docker Compose demo environment
tobira milter         Start the Postfix milter daemon
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/predict` | Classify text as spam/ham |
| POST | `/feedback` | Submit correction feedback |
| GET | `/v1/health` | Health check |
| GET | `/v1/health/ready` | Readiness probe (Kubernetes) |
| GET | `/v1/health/live` | Liveness probe |

## Installation Options

```bash
# Core only
pip install tobira

# With specific backend
pip install tobira[fasttext]
pip install tobira[bert]
pip install tobira[onnx]
pip install tobira[llm]

# Full server setup
pip install tobira[serving,fasttext]

# All features
pip install tobira[serving,fasttext,bert,onnx,llm,evaluation,hub]
```

## License

MIT
