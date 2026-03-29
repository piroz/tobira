# Pricing

tobira is open source and free to use. For organizations that need enterprise-grade features, we offer commercial plans.

## Plans

### Available Now

| | **Community** |
|---|---|
| **Price** | Free |
| **License** | MIT |
| **MTA plugins** | All (rspamd, SpamAssassin, Haraka, Postfix milter) |
| **Inference backends** | All (FastText, BERT, ONNX, Ollama, LLM API, Two-Stage, Ensemble) |
| **CLI tools** | Full (`init`, `doctor`, `monitor`, `train`, `evaluate`, `demo`, `distill`, `hub-push/pull`) |
| **Single-tenant deployment** | :material-check: |
| **Docker / Kubernetes deployment** | :material-check: |
| **Web dashboard** | :material-check: |
| **A/B testing** | :material-check: |
| **Active learning** | :material-check: |
| **Knowledge distillation** | :material-check: |
| **AI-generated text detection** | :material-check: |

### Planned

| | **Enterprise** | **Cloud** |
|---|---|---|
| **Price** | Contact us | Contact us |
| **License** | Commercial | Commercial |
| **Includes all Community features** | :material-check: | :material-check: |
| **Multi-tenant management** | Planned | Planned |
| **RBAC + audit logs** | Planned | Planned |
| **OpenTelemetry integration** | Planned | Planned |
| **Grafana dashboard templates** | Planned | Planned |
| **SSO (SAML / OIDC)** | Planned | Planned |
| **Cloud-based model training** | | Planned |
| **Managed GPU resources** | | Planned |
| **Priority support + SLA** | Planned | Planned |

## Community (Free)

The open-source core includes everything you need to deploy ML-powered spam detection:

- All MTA plugins for major mail servers (rspamd, SpamAssassin, Haraka, Postfix milter)
- All inference backends with full functionality (FastText, BERT, ONNX, Ollama, LLM API, Ensemble)
- CLI-driven guided setup (`tobira init`), diagnostics (`tobira doctor`), and monitoring
- Docker Compose and Kubernetes deployment with health checks
- Web dashboard for real-time monitoring
- GDPR-aware PII anonymization for training data
- A/B testing framework for model comparison
- Active learning with uncertainty-based sampling
- Knowledge distillation (teacher → student model compression)
- AI-generated text detection
- HuggingFace Hub integration for model sharing
- Community support via GitHub Issues and Discussions

[Get Started](getting-started/quickstart.md){ .md-button .md-button--primary }

## Enterprise

For organizations running tobira across multiple teams or domains. All features below are **planned for 2026 H2** — see our [Roadmap](roadmap.md) for details.

- **Multi-tenant management** — Isolated configurations per tenant with centralized administration
- **RBAC + audit logs** — Role-based access control with full audit trail
- **Observability** — OpenTelemetry metrics export and pre-built Grafana dashboards
- **SSO integration** — SAML and OIDC for enterprise identity providers
- **Priority support** — Dedicated support channel with SLA

[Join Waitlist](#waitlist){ .md-button }

## Cloud

Hybrid cloud model — inference stays on-premise, training runs on our GPU infrastructure. All features below are **planned for 2027** — see our [Roadmap](roadmap.md) for details.

- **Cloud-based training** — Upload anonymized data, train models on managed GPUs, download results
- **Managed infrastructure** — No GPU procurement or maintenance required
- Includes all Enterprise features

[Join Waitlist](#waitlist){ .md-button }

---

## Waitlist { #waitlist }

Enterprise and Cloud plans are in the planning stage. Join the waitlist to get early access and help shape the product.

[:material-email-outline: Join the Waitlist](https://forms.gle/6HXu1Xbk13cKYUa39){ .md-button .md-button--primary target="_blank" }

!!! info "What happens next?"
    1. Fill out a short form (email, organization, deployment scale)
    2. We'll reach out to understand your requirements
    3. Early waitlist members get priority access and input on feature prioritization

## FAQ

### Is the Community edition production-ready?

Yes. The Community edition includes all inference backends, MTA plugins, CLI tools, and Docker deployment — everything needed to run tobira in production for a single organization.

### What's the difference between Enterprise and Cloud?

Enterprise is a self-hosted solution with multi-tenant management and enterprise integrations. Cloud adds managed GPU infrastructure for model training — you send anonymized data, we handle the compute.

### Will pricing change?

Pricing details will be finalized based on early adopter feedback. Waitlist members will be notified before any changes and offered founding-member pricing.

### Can I try Enterprise features before committing?

We plan to offer a trial period once Enterprise features are available. Join the waitlist to be notified when trials become available.
