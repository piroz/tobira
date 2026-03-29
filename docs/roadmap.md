# Roadmap

This page outlines planned features for tobira. Timelines are estimates and may change based on community feedback and development priorities.

!!! info "Want to influence priorities?"
    [Join the waitlist](https://forms.gle/6HXu1Xbk13cKYUa39) — early members get priority access and input on feature prioritization.

## Current: Community Edition (v0.5)

The open-source Community edition is **production-ready** and includes:

- :material-check: All inference backends (FastText, BERT, ONNX, Ollama, LLM API, Ensemble, Two-Stage)
- :material-check: All MTA plugins (rspamd, SpamAssassin, Haraka, Postfix milter)
- :material-check: Full CLI toolset (`init`, `doctor`, `monitor`, `train`, `evaluate`, `demo`, `distill`, `hub-push/pull`, `ab-test`, `active-learning`)
- :material-check: A/B testing and active learning
- :material-check: Web dashboard
- :material-check: Knowledge distillation
- :material-check: AI-generated text detection
- :material-check: HuggingFace Hub integration
- :material-check: GDPR-aware PII anonymization
- :material-check: Docker Compose deployment with health checks
- :material-check: Kubernetes-ready health probes (readiness / liveness)
- :material-check: PostgreSQL and Redis storage backends

## Planned: Enterprise Features — Target 2026 H2

These features are in the **planning stage**. No implementation has started yet. Target availability: **second half of 2026**.

| Feature | Description | Status |
|---|---|---|
| **Multi-tenant management** | Isolated configurations per tenant with centralized administration | Planning |
| **RBAC + audit logs** | Role-based access control with full audit trail | Planning |
| **OpenTelemetry integration** | Metrics and traces export via OpenTelemetry protocol | Planning |
| **Grafana dashboard templates** | Pre-built dashboards for prediction monitoring and drift detection | Planning |
| **SSO (SAML / OIDC)** | Enterprise identity provider integration | Planning |
| **Priority support + SLA** | Dedicated support channel with service-level agreement | Planning |

!!! warning "Timeline disclaimer"
    Target dates are estimates based on current plans and may shift depending on community feedback, waitlist demand, and development capacity.

## Planned: Cloud Features — Target 2027

These features depend on Enterprise features and are in early **concept stage**. Target availability: **2027**.

| Feature | Description | Status |
|---|---|---|
| **Cloud-based model training** | Upload anonymized data, train on managed GPUs, download results | Concept |
| **Managed GPU resources** | No GPU procurement or maintenance required | Concept |

## How We Prioritize

Feature priority is determined by:

1. **Waitlist demand** — Features most requested by waitlist members ship first
2. **Community feedback** — GitHub Issues and Discussions influence the backlog
3. **Technical dependencies** — Some features (e.g., Cloud) depend on others (e.g., Enterprise auth layer)

## Stay Updated

- **Waitlist**: [Join here](https://forms.gle/6HXu1Xbk13cKYUa39) for early access notifications
- **GitHub Releases**: Watch the repository for release announcements
- **Discussions**: Participate in [GitHub Discussions](https://github.com/velocitylabo/tobira/discussions) for feature requests
