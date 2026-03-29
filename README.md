# tobira (扉)

**ML-powered email spam detection toolkit with MTA integrations.**

Add ML-based spam classification to your existing mail infrastructure — no ML expertise required, no MTA replacement needed, setup in 3 steps.

[![PyPI version](https://img.shields.io/pypi/v/tobira)](https://pypi.org/project/tobira/)
[![npm version](https://img.shields.io/npm/v/tobira)](https://www.npmjs.com/package/tobira)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-1%2C200%2B%20passed-brightgreen)](https://github.com/velocitylabo/tobira)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

## 3 Steps to Get Started

```bash
pip install tobira[serving,fasttext]
tobira init        # Auto-detect your MTA, generate config
tobira serve       # Start API server — done
```

Your MTA plugin talks to the tobira API over HTTP. No changes to your mail flow.

```
MTA (Postfix, etc.)
  └── MTA Plugin (rspamd / SpamAssassin / Haraka / milter)
        └── HTTP POST /predict → tobira API Server
```

## Why tobira

- **Works with your existing MTA** — Plugins for rspamd, SpamAssassin, Haraka, and Postfix milter. No migration required.
- **No ML expertise needed** — `tobira init` detects your MTA and generates config. `tobira doctor` validates your setup.
- **7 inference backends** — From FastText (1ms, CPU) to BERT (highest accuracy) to LLM APIs (zero training data). Swap backends without changing your setup.
- **Production-ready** — Fail-open mode, health checks, Docker Compose / Kubernetes deployment. 1,200+ tests.

<details>
<summary>All features</summary>

- **A/B testing** — Compare models in production with random or hash-based traffic splitting
- **Active learning** — Uncertainty-based sampling to prioritize labeling effort
- **Web dashboard** — Real-time prediction stats, score distribution, and drift visualization
- **Knowledge distillation** — Compress large teacher models into lightweight student models
- **AI-generated text detection** — Heuristic-based detection of machine-generated content
- **HuggingFace Hub** — Push and pull models with auto-generated model cards
- **GDPR-aware** — PII anonymization with regex + NER (GiNZA) for training data
- **Monitoring** — Score drift detection (PSI/KS test), automatic retraining triggers

</details>

## Documentation

**[velocitylabo.github.io/tobira](https://velocitylabo.github.io/tobira)** — Full documentation:

- [Quick Start](https://velocitylabo.github.io/tobira/getting-started/quickstart/) — Install and run in 5 minutes
- [MTA Tutorials](https://velocitylabo.github.io/tobira/mta/) — Step-by-step integration guides
- [CLI Reference](https://velocitylabo.github.io/tobira/cli/) — All CLI subcommands
- [API Reference](https://velocitylabo.github.io/tobira/api/) — HTTP endpoint documentation
- [Deployment Guide](https://velocitylabo.github.io/tobira/getting-started/phased-rollout/) — Phased rollout strategy
- [Pricing](https://velocitylabo.github.io/tobira/pricing/) — Community (free), Enterprise, and Cloud plans
- [Roadmap](https://velocitylabo.github.io/tobira/roadmap/) — Planned features and timeline

## Packages

| Package | Registry | Description |
|---------|----------|-------------|
| `tobira` | [PyPI](https://pypi.org/project/tobira/) | Python toolkit (backends, API server, CLI) |
| `tobira` | [npm](https://www.npmjs.com/package/tobira) | JavaScript client for Node.js / Haraka |

## Deployment

| Method | Path | Description |
|--------|------|-------------|
| Docker Compose | [`docker/`](docker/) | Full test environment with all MTAs |
| Helm Chart | [`charts/tobira/`](charts/tobira/) | Kubernetes deployment with HPA/Ingress |
| Plain K8s | [`k8s/`](k8s/) | Kubernetes manifests without Helm |

## License

MIT
