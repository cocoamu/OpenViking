# OpenViking Helm Chart

Deploy OpenViking on Kubernetes using Helm.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.x
- A storage class that supports `ReadWriteOnce` persistent volumes (for RocksDB data)

## Installation

### Quick Start

```bash
helm install openviking ./deploy/helm/openviking \
  --set-string config.server.root_api_key="YOUR_ROOT_API_KEY" \
  --set-string config.embedding.dense.api_key="YOUR_VOLCENGINE_API_KEY" \
  --set-string config.vlm.api_key="YOUR_VOLCENGINE_API_KEY"
```

The chart deploys `ghcr.io/volcengine/openviking:latest` by default. To choose
a different image tag:

```bash
# newest image from the main branch
helm upgrade --install openviking ./deploy/helm/openviking --set image.tag=main

# pinned release image
helm upgrade --install openviking ./deploy/helm/openviking --set image.tag=v0.3.17

# immutable image reference (digest takes precedence over tag)
helm upgrade --install openviking ./deploy/helm/openviking \
  --set image.repository=registry.example/openviking \
  --set image.digest=sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

# use the default latest image
helm upgrade --install openviking ./deploy/helm/openviking --set image.tag=
```

We recommend the `ghcr.io` image. If `ghcr.io` is hard to reach, add
`--set image.repository=openviking-cn-beijing.cr.volces.com/volcengine/openviking`.

### Install with Custom Values

Create a `my-values.yaml` file:

```yaml
replicaCount: 1

resources:
  limits:
    cpu: "4"
    memory: 8Gi
  requests:
    cpu: "1"
    memory: 2Gi

persistence:
  size: 50Gi
  storageClass: "gp3"

config:
  storage:
    workspace: /app/.openviking/openviking_workspace
  log:
    level: INFO
    output: stdout
  server:
    host: "0.0.0.0"
    port: 1933
    workers: 1
    root_api_key: "your-secret-key"
  embedding:
    dense:
      api_base: "https://ark.cn-beijing.volces.com/api/v3"
      api_key: "your-volcengine-api-key"
      provider: "volcengine"
      dimension: 1024
      model: "doubao-embedding-vision-251215"
      input: "multimodal"
    max_concurrent: 10
  vlm:
    api_base: "https://ark.cn-beijing.volces.com/api/v3"
    api_key: "your-volcengine-api-key"
    provider: "volcengine"
    model: "doubao-seed-2-0-lite-260428"
    temperature: 0.0
    max_retries: 2
    thinking: false
    max_concurrent: 100
```

Then install:

```bash
helm install openviking ./deploy/helm/openviking -f my-values.yaml
```

### Using Secrets for API Keys

For production, avoid putting API keys directly in values. Use `extraEnv` with
Kubernetes secrets instead:

```bash
# Create a secret
kubectl create secret generic openviking-api-keys \
  --from-literal=root-api-key="YOUR_ROOT_API_KEY" \
  --from-literal=embedding-api-key="YOUR_KEY" \
  --from-literal=vlm-api-key="YOUR_KEY"
```

Then reference it in your values:

```yaml
config:
  server:
    root_api_key: "${OPENVIKING_ROOT_API_KEY}"
  embedding:
    dense:
      api_key: "${OPENVIKING_EMBEDDING_API_KEY}"
  vlm:
    api_key: "${OPENVIKING_VLM_API_KEY}"

extraEnv:
  - name: OPENVIKING_ROOT_API_KEY
    valueFrom:
      secretKeyRef:
        name: openviking-api-keys
        key: root-api-key
  - name: OPENVIKING_EMBEDDING_API_KEY
    valueFrom:
      secretKeyRef:
        name: openviking-api-keys
        key: embedding-api-key
  - name: OPENVIKING_VLM_API_KEY
    valueFrom:
      secretKeyRef:
        name: openviking-api-keys
        key: vlm-api-key
```

OpenViking expands environment variables inside `ov.conf` at startup, so the
ConfigMap can contain placeholders while the actual secrets stay in Kubernetes
Secrets.

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image repository | `ghcr.io/volcengine/openviking` |
| `image.tag` | Container image tag (`latest`, `main`, pinned release, or empty for `latest`) | `latest` |
| `image.digest` | Immutable image digest; rendered as `repository@digest` when set | `""` |
| `image.pullPolicy` | Image pull policy | `Always` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `1933` |
| `persistence.enabled` | Enable persistent storage | `true` |
| `persistence.size` | PVC size | `20Gi` |
| `persistence.storageClass` | Storage class name | `""` (default) |
| `persistence.mountPath` | Container path for OpenViking persistent state | `/app/.openviking` |
| `bot.enabled` | Start vikingbot alongside the API server | `false` |
| `persistence.existingClaim` | Use an existing PVC | `""` |
| `resources.limits.cpu` | CPU limit | `2` |
| `resources.limits.memory` | Memory limit | `4Gi` |
| `resources.requests.cpu` | CPU request | `500m` |
| `resources.requests.memory` | Memory request | `1Gi` |
| `ingress.enabled` | Enable ingress | `false` |
| `config.server.root_api_key` | API key required when server binds to 0.0.0.0 | `""` |
| `config` | Full ov.conf configuration object | See `values.yaml` |
| `extraEnv` | Additional environment variables | `[]` |

## Upgrading

```bash
helm upgrade openviking ./deploy/helm/openviking -f my-values.yaml
```

The deployment uses a `Recreate` strategy to avoid data corruption from
multiple pods accessing the same RocksDB volume simultaneously.

## Reliable two-replica mode

Two replicas are supported only when the pod-local workspace is disposable and
all state needed for task ownership is shared outside the pod. Configure the
OpenViking file backend and vector database for their shared production
services, then enable the Redis-backed QueueFS and PathLock settings:

```yaml
replicaCount: 2

strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 0
    maxSurge: 1

# Each pod gets an independent emptyDir workspace. A shared RWO PVC is unsafe.
persistence:
  enabled: false

config:
  cache:
    provider: redis
    params:
      endpoints: ["redis://<verified-durable-redis-service>:6379"]
      command_timeout_ms: 1000
  storage:
    workspace: /app/.openviking
    distributed_task_writes: true
    agfs:
      queuefs:
        backend: cache
        mode: shared
        cache_key_prefix: prod-openviking
      pathlock:
        provider: cache
        namespace: prod-openviking
        lock_expire_secs: 60
```

The chart automatically adds required node anti-affinity when
`replicaCount > 1`. Before switching a production Service, verify that the
external AGFS file store, VectorDB, and Redis use durable/shared storage and
that both pods can read the same task records. The chart refuses
`replicaCount > 1` while `persistence.enabled=true` so a release cannot mount
the default per-release RWO workspace from two pods. It also refuses a local AGFS
or local VectorDB backend in two-replica mode, because those stores would diverge
between pods even when task state is shared.

## Uninstalling

```bash
helm uninstall openviking
```

Note: The PersistentVolumeClaim is retained by Helm and becomes orphaned after
uninstall. Delete it explicitly to remove the stored data:

```bash
kubectl delete pvc openviking-data
```
