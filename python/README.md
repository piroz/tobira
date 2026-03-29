# tobira

**ML-powered spam detection toolkit for mail servers.**

Add ML-based spam classification to your existing mail infrastructure — rspamd, SpamAssassin, Haraka, or Postfix — without ML expertise.

[![PyPI version](https://img.shields.io/pypi/v/tobira)](https://pypi.org/project/tobira/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1%2C200%2B%20passed-brightgreen)](https://github.com/velocitylabo/tobira)

**[Documentation](https://velocitylabo.github.io/tobira)** | **[GitHub](https://github.com/velocitylabo/tobira)** | **[Roadmap](https://velocitylabo.github.io/tobira/roadmap/)**

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

## Backends

| Backend | Use case | Latency |
|---------|----------|---------|
| **FastText** | High-throughput / low-resource | ~1ms |
| **BERT** | Highest accuracy | ~50ms (GPU) |
| **ONNX** | Fast + accurate (quantized) | ~30ms (CPU) |
| **LLM API** | Zero training data needed | Varies |
| **Ollama** | Privacy-first / air-gapped | Varies |
| **Two-Stage** | FastText screening → BERT precision | ~5ms avg |
| **Ensemble** | Maximum reliability | Varies |

Swap backends by changing one line in `tobira.toml`. Custom backends can be added via the `tobira.backends` entry point.

## MTA Integration

Ready-to-use plugins — all support fail-open mode and tiered spam scoring:

| MTA | Plugin | Config file |
|-----|--------|-------------|
| rspamd | `tobira.lua` | `tobira.conf` |
| SpamAssassin | `Tobira.pm` | `tobira.cf` |
| Haraka | Node.js plugin | `tobira.ini` |
| Postfix | milter (`tobira milter`) | `tobira-milter.conf` |

See [MTA Tutorials](https://velocitylabo.github.io/tobira/mta/) for step-by-step guides.

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

## CLI Commands

```
tobira init           Detect MTA and generate config
tobira serve          Start the prediction API server
tobira doctor         Verify config, backends, and connectivity
tobira demo           Launch a Docker Compose demo environment
tobira train          Fine-tune a model on your labeled data
tobira evaluate       Evaluate model accuracy on a test set
tobira monitor        Analyze prediction logs for drift (PSI, KS)
tobira distill        Knowledge distillation (teacher → student)
tobira hub-push/pull  Share models via Hugging Face Hub
tobira milter         Start the Postfix milter daemon
```

## Documentation

Full documentation at **[velocitylabo.github.io/tobira](https://velocitylabo.github.io/tobira)** — MTA integration tutorials, CLI reference, API reference, deployment guides, and pricing.

## License

MIT
