# Pricing

tobira is open source and free to use. For organizations that need enterprise-grade features, we offer commercial plans.

## Plans

| | **Community** | **Enterprise** | **Cloud** |
|---|---|---|---|
| **Price** | Free | Contact us | Contact us |
| **License** | MIT | Commercial | Commercial |
| **Inference backends** | All (FastText, BERT, ONNX, Ollama, LLM API, Ensemble) | All | All |
| **MTA plugins** | All (rspamd, SpamAssassin, Haraka, Postfix milter) | All | All |
| **CLI tools** | Full (`init`, `doctor`, `monitor`, `train`, `evaluate`, `demo`) | Full | Full |
| **Single-tenant deployment** | :material-check: | :material-check: | :material-check: |
| **Multi-tenant management** | | :material-check: | :material-check: |
| **RBAC + audit logs** | | :material-check: | :material-check: |
| **OpenTelemetry integration** | | :material-check: | :material-check: |
| **Grafana dashboard templates** | | :material-check: | :material-check: |
| **SSO (SAML / OIDC)** | | :material-check: | :material-check: |
| **Cloud-based model training** | | | :material-check: |
| **Managed GPU resources** | | | :material-check: |
| **Priority support + SLA** | | :material-check: | :material-check: |

## Community (Free)

The open-source core includes everything you need to deploy ML-powered spam detection:

- All inference backends with full functionality
- All MTA plugins for major mail servers
- CLI-driven guided setup, diagnostics, and monitoring
- GDPR-aware PII anonymization for training data
- Docker Compose deployment with health checks
- Community support via GitHub Issues and Discussions

[Get Started](getting-started/quickstart.md){ .md-button .md-button--primary }

## Enterprise

For organizations running tobira across multiple teams or domains:

- **Multi-tenant management** — Isolated configurations per tenant with centralized administration
- **RBAC + audit logs** — Role-based access control with full audit trail
- **Observability** — OpenTelemetry metrics export and pre-built Grafana dashboards
- **SSO integration** — SAML and OIDC for enterprise identity providers
- **Priority support** — Dedicated support channel with SLA

[Join Waitlist](#waitlist){ .md-button }

## Cloud

Hybrid cloud model — inference stays on-premise, training runs on our GPU infrastructure:

- **Cloud-based training** — Upload anonymized data, train models on managed GPUs, download results
- **Managed infrastructure** — No GPU procurement or maintenance required
- Includes all Enterprise features

[Join Waitlist](#waitlist){ .md-button }

---

## Waitlist { #waitlist }

Enterprise and Cloud plans are currently in development. Join the waitlist to get early access and help shape the product.

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

We plan to offer a trial period for Enterprise features. Join the waitlist to be notified when trials become available.
