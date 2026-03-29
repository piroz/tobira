# Plain Kubernetes Manifests

Kubernetes manifests for deploying tobira without Helm.

## Usage

```bash
kubectl apply -f k8s/
```

## Customization

Edit the YAML files directly to match your environment:

- `configmap.yaml` — Backend type and model path
- `deployment.yaml` — Replicas, resources, image tag
- `pvc.yaml` — Storage size and class
- `service.yaml` — Service type and port

For a more flexible deployment, use the [Helm chart](../charts/tobira/).

## Documentation

See the [Deployment Guide](https://velocitylabo.github.io/tobira/getting-started/phased-rollout/) for phased rollout strategies.
