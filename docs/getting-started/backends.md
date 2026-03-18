# Backends

tobira supports multiple inference backends through the `BackendProtocol` abstraction. Choose based on your accuracy, latency, and resource requirements.

## Comparison

| Backend | Model Size | Latency | Hardware | Accuracy |
|---------|-----------|---------|----------|----------|
| **FastText** | ~10 MB | ~1 ms | CPU only | Medium |
| **ONNX** | ~110 MB (quantized) | ~30 ms | CPU only | High |
| **BERT** | ~440 MB | ~200 ms | GPU recommended | High |
| **Ollama** | 1-70 GB | ~500 ms (GPU) | GPU recommended | High |
| **LLM API** | Remote | ~300 ms | Network | Highest |
| **Ensemble** | Varies | Varies | Varies | Highest |
| **Two-Stage** | Combined | ~1-30 ms | CPU | High |

## FastText

Lightweight n-gram model. Best for high-throughput environments where speed matters most.

```toml
[backend]
type = "fasttext"
model_path = "/var/lib/tobira/fasttext-spam.bin"
```

**When to use**: First-stage filter, resource-constrained environments, very high email volumes.

## ONNX

Quantized BERT model running on ONNX Runtime. Best balance of accuracy and speed on CPU.

```toml
[backend]
type = "onnx"
model_path = "/var/lib/tobira/model_int8.onnx"
model_name = "tohoku-nlp/bert-base-japanese-v3"
```

**When to use**: Production deployments on CPU-only servers, best accuracy-to-cost ratio.

## BERT (PyTorch)

Full BERT model via HuggingFace Transformers. Highest accuracy for fine-tuned models.

```toml
[backend]
type = "bert"
model_name = "tohoku-nlp/bert-base-japanese-v3"
device = "cuda"
```

**When to use**: GPU-equipped servers, when maximum fine-tuned accuracy is needed.

## Ollama

Local LLM inference via Ollama. Supports any Ollama-compatible model.

```toml
[backend]
type = "ollama"
model = "llama3"
base_url = "http://localhost:11434"
timeout = 30
```

**When to use**: When you want LLM-level understanding without external API calls.

## LLM API

Cloud LLM via OpenAI-compatible API. Works with OpenAI, Anthropic (via proxy), and other providers.

```toml
[backend]
type = "llm_api"
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
timeout = 30
```

**When to use**: Highest accuracy needed, acceptable latency and cost, data privacy allows cloud processing.

## Ensemble

Combines multiple backends using weighted average or majority vote.

```toml
[backend]
type = "ensemble"
strategy = "weighted_average"
weights = [0.3, 0.7]

[[backend.backends]]
type = "fasttext"
model_path = "/var/lib/tobira/fasttext-spam.bin"

[[backend.backends]]
type = "onnx"
model_path = "/var/lib/tobira/model_int8.onnx"
```

**When to use**: Maximum accuracy through model agreement, critical environments.

## Two-Stage Filter

Routes emails through a fast first-stage model, sending only uncertain results to a precise second-stage model. Reduces load on the expensive model by 80-90%.

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

**When to use**: High-volume environments where you want both speed and accuracy.

The grey zone defines the score range where the first-stage result is uncertain:

- Score < 0.3 (low): Classified as ham by first stage, no second stage needed
- Score > 0.7 (high): Classified as spam by first stage, no second stage needed
- 0.3 <= Score <= 0.7: Grey zone, forwarded to second stage for precise classification
