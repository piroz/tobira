# tobira (扉)

**ML-powered email spam detection toolkit with MTA integrations.**

tobira bridges modern ML models (BERT, ONNX, LLMs) with existing mail transfer agents (rspamd, SpamAssassin, Haraka, Postfix). Mail administrators can add ML-based spam detection as a plugin — no ML expertise required.

## Key Features

- **4 MTA plugins** — rspamd (Lua), SpamAssassin (Perl), Haraka (Node.js), Postfix milter (Python). No MTA replacement needed.
- **No ML expertise required** — `tobira init` detects your MTA and generates config. `tobira doctor` validates your setup.
- **7 inference backends** — FastText, BERT, ONNX, Ollama, LLM API, Ensemble, and Two-Stage filtering. Swap backends without changing your setup.
- **Production-ready** — Docker Compose / Kubernetes deployment, health checks, fail-open mode. 1,200+ tests.
- **A/B testing** — Compare models in production with random or hash-based traffic splitting
- **Active learning** — Uncertainty-based sampling to prioritize labeling effort
- **Web dashboard** — Real-time prediction stats, score distribution, and drift visualization
- **Knowledge distillation** — Compress large teacher models into lightweight student models
- **AI-generated text detection** — Heuristic-based detection of machine-generated content
- **HuggingFace Hub integration** — Push and pull models with auto-generated model cards
- **GDPR-aware** — PII anonymization with regex + NER (GiNZA) for training data

## Architecture

```
MTA (Postfix, etc.)
  └── MTA Plugin (rspamd / SpamAssassin / Haraka / milter)
        └── HTTP POST /predict
              └── tobira API Server (FastAPI)
                    └── BackendProtocol
                          ├── FastTextBackend
                          ├── BertBackend
                          ├── OnnxBackend
                          ├── OllamaBackend
                          ├── LlmApiBackend
                          ├── TwoStageBackend
                          └── EnsembleBackend
```

## Quick Links

- [Quick Start](getting-started/quickstart.md) — Install and run in 5 minutes
- [MTA Tutorials](mta/index.md) — Step-by-step integration guides
- [CLI Reference](cli.md) — All CLI subcommands
- [API Reference](api.md) — HTTP endpoint documentation
- [Deployment Guide](getting-started/phased-rollout.md) — Phased rollout strategy
- [Pricing](pricing.md) — Community, Enterprise, and Cloud plans
- [Roadmap](roadmap.md) — Planned features and timeline

## Installation

=== "Python (PyPI)"

    ```bash
    pip install tobira
    ```

=== "With extras"

    ```bash
    # FastText backend
    pip install tobira[fasttext]

    # BERT backend
    pip install tobira[bert]

    # ONNX backend
    pip install tobira[onnx]

    # API server
    pip install tobira[serving]

    # All features
    pip install tobira[fasttext,bert,onnx,serving,evaluation,llm]
    ```

=== "npm (Haraka plugin)"

    ```bash
    npm install tobira
    ```
