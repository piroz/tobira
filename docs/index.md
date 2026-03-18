# tobira (扉)

**ML-powered email spam detection toolkit with MTA integrations.**

tobira bridges modern ML models (BERT, ONNX, LLMs) with existing mail transfer agents (rspamd, SpamAssassin, Haraka, Postfix). Mail administrators can add ML-based spam detection as a plugin — no ML expertise required.

## Key Features

- **Multiple inference backends** — FastText, BERT, ONNX, Ollama, LLM API, ensemble, and two-stage filtering
- **MTA plugins** — rspamd (Lua), SpamAssassin (Perl), Haraka (Node.js), Postfix milter (Python)
- **CLI tools** — `tobira init` for guided setup, `tobira doctor` for diagnostics, `tobira monitor` for drift detection
- **GDPR-aware** — PII anonymization with regex + NER (GiNZA) for training data
- **Production-ready** — Docker Compose deployment, health checks, fail-open mode

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
                          └── EnsembleBackend
```

## Quick Links

- [Quick Start](getting-started/quickstart.md) — Install and run in 5 minutes
- [MTA Tutorials](mta/index.md) — Step-by-step integration guides
- [CLI Reference](cli.md) — All CLI subcommands
- [API Reference](api.md) — HTTP endpoint documentation
- [Deployment Guide](getting-started/phased-rollout.md) — Phased rollout strategy

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
