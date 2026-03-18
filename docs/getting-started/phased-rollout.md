# Phased Deployment Guide

tobira is designed for zero-downtime integration with your existing mail infrastructure. This guide describes a phased rollout strategy.

## Overview

```
Phase A: MTA standard filters (immediate)
    ↓
Phase B: ML model training (parallel)
    ↓
Phase C: ML model deployment (zero-downtime switch)
    ↓
Phase D: Continuous improvement
```

## Phase A: MTA Standard Filters

Start with your MTA's built-in filtering (Bayesian, SPF, DKIM, fuzzy hashing, etc.). Install the tobira plugin but keep it disabled.

**Goal**: Begin collecting labeled spam/ham data for ML training.

```bash
# Install tobira and configure for your MTA
tobira init
```

`tobira init` auto-detects your running MTA and generates the appropriate plugin configuration.

**Duration**: Immediate. Start collecting data from day one.

## Phase B: ML Model Training

While Phase A runs in production, train your ML model on the collected data.

1. Anonymize training data (PII removal):
    ```bash
    # tobira handles PII anonymization automatically during training
    ```

2. Train and evaluate:
    ```bash
    tobira evaluate --data /path/to/labeled-data.jsonl --backend onnx
    ```

3. Check if the model meets quality thresholds:
    ```bash
    tobira monitor --check-threshold --target-f1 0.95
    ```

**Duration**: Hours to days, depending on data volume and model complexity.

**Requirement**: F1 score > 0.95 before proceeding to Phase C.

## Phase C: ML Model Deployment

Deploy the trained model with zero downtime.

1. Start the tobira API server:
    ```bash
    tobira serve --config /etc/tobira/tobira.toml
    ```

2. Verify the server is healthy:
    ```bash
    tobira doctor
    ```

3. Enable the MTA plugin:

    === "rspamd"

        Edit `/etc/rspamd/local.d/tobira.conf` and reload:
        ```bash
        rspamadm configtest && systemctl reload rspamd
        ```

    === "SpamAssassin"

        Enable the plugin in `local.cf` and restart:
        ```bash
        systemctl restart spamassassin
        ```

    === "Haraka"

        Add `tobira` to `config/plugins` and restart:
        ```bash
        systemctl restart haraka
        ```

    === "Postfix milter"

        Start the milter daemon:
        ```bash
        tobira milter --config /etc/tobira/milter.conf
        ```

**Duration**: Configuration change only. No mail service interruption.

## Phase D: Continuous Improvement

Monitor model performance and retrain as needed.

```bash
# One-shot analysis
tobira monitor

# Continuous monitoring (daemon mode)
tobira monitor --watch
```

tobira's monitoring detects:

- **Concept drift** — Score distribution shifts over time
- **Threshold tuning** — Optimal classification thresholds
- **Backend upgrade suggestions** — When to switch to a more accurate backend

## Infrastructure Requirements

| Phase | Infrastructure | Data |
|-------|---------------|------|
| A | MTA filter (rspamd / SpamAssassin / Haraka) | None (start collecting) |
| B | GPU environment (for training) | 500+ labeled emails |
| C | API server (CPU is sufficient) | Trained model |
| D | Same as Phase C | Continuously collected data |
