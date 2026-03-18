# tobira Helm Chart

A Helm chart for deploying the tobira spam prediction API server on Kubernetes.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.x
- A container image with tobira installed

## Installation

```bash
helm install tobira charts/tobira/
```

### With custom values

```bash
helm install tobira charts/tobira/ \
  --set backend.type=onnx \
  --set backend.modelPath=/models/model_int8.onnx \
  --set replicaCount=3
```

### Using an existing PVC for models

```bash
helm install tobira charts/tobira/ \
  --set persistence.existingClaim=my-model-pvc
```

## Configuration

See [values.yaml](values.yaml) for the full list of configurable parameters.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image repository | `tobira` |
| `image.tag` | Container image tag | Chart appVersion |
| `backend.type` | Backend type | `fasttext` |
| `backend.modelPath` | Path to model file | `/models/fasttext-spam.bin` |
| `persistence.enabled` | Enable PVC for models | `true` |
| `persistence.size` | PVC size | `5Gi` |
| `autoscaling.enabled` | Enable HPA | `false` |
| `ingress.enabled` | Enable Ingress | `false` |

## Plain Kubernetes Manifests

For users who prefer not to use Helm, plain Kubernetes manifests are available in the `k8s/` directory at the repository root.

```bash
kubectl apply -f k8s/
```
