# Configuration

tobira uses TOML configuration files. Pass the config file path with `--config`:

```bash
tobira serve --config /etc/tobira/tobira.toml
```

## Configuration File Structure

```toml
[backend]
type = "fasttext"           # Backend type (required)
model_path = "/var/lib/tobira/model.bin"

[monitoring]
enabled = true
store_type = "jsonl"        # "jsonl" or "redis"
log_path = "/var/log/tobira/predictions.jsonl"

[monitoring.redis]
url = "redis://localhost:6379"
prefix = "tobira:"

[feedback]
enabled = true
store_path = "/var/lib/tobira/feedback.jsonl"

[header_analysis]
enabled = false             # Enable email header risk scoring

[dashboard]
enabled = false             # Enable web dashboard at /dashboard

[ai_detection]
enabled = false             # Enable AI-generated text detection
```

## Backend Configuration

Each backend type has its own configuration options. See [Backends](backends.md) for details.

### FastText

```toml
[backend]
type = "fasttext"
model_path = "/var/lib/tobira/fasttext-spam.bin"
```

### BERT (PyTorch)

```toml
[backend]
type = "bert"
model_name = "tohoku-nlp/bert-base-japanese-v3"
device = "cuda"  # or "cpu"
```

### ONNX

```toml
[backend]
type = "onnx"
model_path = "/var/lib/tobira/model_int8.onnx"
model_name = "tohoku-nlp/bert-base-japanese-v3"
```

### Two-Stage Filter

```toml
[backend]
type = "two_stage"

[backend.first_stage]
type = "fasttext"
model_path = "/var/lib/tobira/fasttext-spam.bin"

[backend.second_stage]
type = "onnx"
model_path = "/var/lib/tobira/model_int8.onnx"

[backend.grey_zone]
low = 0.3
high = 0.7
```

### Ensemble

```toml
[backend]
type = "ensemble"
strategy = "weighted_average"  # or "majority_vote"
weights = [0.3, 0.7]

[[backend.backends]]
type = "fasttext"
model_path = "/var/lib/tobira/fasttext-spam.bin"

[[backend.backends]]
type = "onnx"
model_path = "/var/lib/tobira/model_int8.onnx"
```

## Environment Variables

All configuration values can be overridden with environment variables using the `TOBIRA_` prefix:

| Variable | Description |
|----------|-------------|
| `TOBIRA_BACKEND` | Backend type |
| `TOBIRA_BACKEND_CONFIG` | Path to backend-specific config |
| `TOBIRA_HOST` | Server bind host (default: `127.0.0.1`) |
| `TOBIRA_PORT` | Server bind port (default: `8000`) |
