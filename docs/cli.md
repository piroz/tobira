# CLI Reference

The `tobira` command provides subcommands for server management, diagnostics, evaluation, and monitoring.

## Installation

```bash
pip install tobira[serving]
```

## Subcommands

### `tobira serve`

Start the FastAPI API server.

```bash
tobira serve [--config CONFIG] [--host HOST] [--port PORT] [--backend BACKEND]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | — | Path to TOML configuration file |
| `--host` | `127.0.0.1` | Server bind address |
| `--port` | `8000` | Server bind port |
| `--backend` | — | Backend type (overrides config) |
| `--model-path` | — | Model file path (overrides config) |

### `tobira init`

Auto-detect your MTA and generate integration configuration files.

```bash
tobira init [--mta MTA] [--output-dir DIR]
```

The setup wizard:

1. Detects running MTAs (rspamd, SpamAssassin, Haraka, Postfix)
2. Generates MTA-specific plugin configuration
3. Provides step-by-step installation instructions

### `tobira doctor`

Run diagnostic checks on your tobira installation.

```bash
tobira doctor [--config CONFIG]
```

Checks performed:

- Configuration file validity
- Backend model loading
- API server connectivity
- MTA plugin activation status

### `tobira evaluate`

Evaluate a backend against labeled test data.

```bash
tobira evaluate --data DATA [--backend BACKEND] [--config CONFIG]
```

| Option | Description |
|--------|-------------|
| `--data` | Path to labeled data file (JSONL) |
| `--backend` | Backend type to evaluate |
| `--config` | Configuration file path |

Outputs: accuracy, precision, recall, F1 score, confusion matrix, and PR curve.

### `tobira monitor`

Analyze prediction metrics and suggest improvements.

```bash
tobira monitor [--config CONFIG] [--watch] [--interval SECONDS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | — | Configuration file path |
| `--watch` | — | Enable daemon mode (continuous monitoring) |
| `--interval` | `3600` | Check interval in seconds (daemon mode) |

Analysis includes:

- Score distribution shift detection (PSI / KS test)
- FP/FN rate trends
- Threshold tuning suggestions
- Backend upgrade recommendations

### `tobira hub-push` / `tobira hub-pull`

Manage models on HuggingFace Hub.

```bash
# Push a model to HuggingFace Hub
tobira hub-push --model-path MODEL --repo REPO

# Pull a model from HuggingFace Hub
tobira hub-pull --repo REPO --output-dir DIR
```

### `tobira milter`

Start the Postfix milter daemon.

```bash
tobira milter --config CONFIG
```

See [Postfix milter integration](mta/postfix-milter.md) for configuration details.

### `tobira distill`

Run knowledge distillation from a teacher model to a student model.

```bash
tobira distill --teacher TEACHER --student STUDENT --data DATA
```

| Option | Description |
|--------|-------------|
| `--teacher` | Teacher model path or name |
| `--student` | Student model path or name |
| `--data` | Training data path |
