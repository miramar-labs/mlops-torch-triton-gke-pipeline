# mlops-torch-triton-gke-pipeline

GPU ML training pipeline: fine-tune DistilBERT for text classification on a DGX Station, track experiments with MLflow, serve the model on GKE via Triton Inference Server.

[![ML Train Test](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-train-test.yaml/badge.svg)](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-train-test.yaml)
[![ML Train](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-train.yaml/badge.svg)](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-train.yaml)
[![ML Deploy](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-deploy.yaml/badge.svg)](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-deploy.yaml)

## Links

- **[GCP Artifact Registry](https://console.cloud.google.com/artifacts/docker/miramar-platform/us-west1/apps?project=miramar-platform)** — `us-west1-docker.pkg.dev/miramar-platform/apps/triton-text-classifier`
- **[GKE Workloads](https://console.cloud.google.com/kubernetes/workload/overview?project=miramar-platform)** — `triton` deployment in namespace `mlops-torch-triton-gke-pipeline` on `miramar-shared-gke`
- **[GitHub Actions](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions)** — workflow run history
- **[MLflow UI](http://localhost:5000)** — experiment tracking on DGX; requires SSH tunnel: `ssh -L 5000:localhost:5000 aaron@spark-79b7.local`

## Local Development

```bash
# Create a pyenv virtualenv for this workspace (once)
pyenv virtualenv 3.13.0 pyTriton
pyenv local pyTriton

# Install dependencies (CPU torch — enough for tests and linting)
pip install -r ml/requirements.txt
```

VS Code will use the `pyTriton` interpreter automatically via `.python-version`.

## Pipeline

```
ML Train Test — workflow_dispatch or push to ml/{train.py,test_train.py,Dockerfile.train} (dgx, ARM64)
  ├── pytest → test_train.py  (aborts chain on failure; deps pre-installed in mlabs-runner)
  └── docker build → ml-trainer image

ML Train — triggered by ML Train Test success (dgx, ARM64, GPU)
  ├── docker build → ml-trainer image
  ├── docker run --gpus all → DistilBERT fine-tune on IMDB
  ├── log metrics → MLflow (localhost:5000, via --network host)
  ├── export → model.onnx
  └── upload artifact → onnx-model

ML Deploy — triggered by ML Train success (wsl2, x86_64)
  ├── download artifact → model.onnx
  ├── docker build → Triton serving image (model baked in)
  ├── push → GAR (latest + SHA tag)
  └── kubectl apply → GKE namespace mlops-torch-triton-gke-pipeline
```

## Workflows

| Workflow | File | Runner | Trigger |
|---|---|---|---|
| **ML Train Test** | `ml-train-test.yaml` | `dgx` | Push to `ml/train.py`, `test_train.py`, or `Dockerfile.train`; or manual |
| **ML Train** | `ml-train.yaml` | `dgx` | Auto on ML Train Test success; or manual |
| **ML Deploy** | `ml-deploy.yaml` | `wsl2` | Auto on ML Train success |
| **GKE Cluster Expand** | [`miramar-platform-gcp`](https://github.com/miramar-labs-org/miramar-platform-gcp/actions/workflows/gke-cluster-expand.yaml) | `wsl2` | Manual only |
| **GKE Cluster Restore** | [`miramar-platform-gcp`](https://github.com/miramar-labs-org/miramar-platform-gcp/actions/workflows/gke-cluster-restore.yaml) | `wsl2` | Manual only |

### Testing Triton when the cluster has insufficient resources

Run these three workflows in sequence to temporarily scale up the cluster, deploy, and restore:

1. **GKE Cluster Expand** — saves current node pool state (count + machine type) to GCS, then resizes the pool up
2. **ML Deploy** — trigger manually to build and deploy the Triton serving image
3. **GKE Cluster Restore** — reads the saved state from GCS and restores the pool automatically (optional `node_count_override` if needed)

### ML Train inputs (manual dispatch only)

| Input | Default | Description |
|---|---|---|
| `epochs` | `3` | Number of training epochs |
| `experiment` | `text-classifier` | MLflow experiment name |

The model artifact (`onnx-model`) passes between workflows via GitHub Actions artifact storage. The deploy workflow uses the training run's commit SHA as the image tag, so `latest` and the SHA-tagged image in GAR always correspond to the same trained model.

## Model

| Property | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Task | Binary sentiment classification (IMDB) |
| Dataset | HuggingFace `datasets` — `imdb` (25k train / 25k test) |
| Max sequence length | 128 tokens |
| Export format | ONNX (opset 14) |
| Inference backend | Triton ONNX Runtime |

## GCP / GKE

| Resource | Value |
|---|---|
| Project | `miramar-platform` |
| Cluster | `miramar-shared-gke` (`us-west1-a`) |
| Namespace | `mlops-torch-triton-gke-pipeline` |
| Artifact Registry | `us-west1-docker.pkg.dev/miramar-platform/apps/triton-text-classifier` |
| Auth | Workload Identity Federation — no long-lived keys |

## Runners

Two self-hosted runners are required. The runner image (`mlabs-runner`) and launch scripts live in [miramar-platform-gcp](https://github.com/miramar-labs-org/miramar-platform-gcp).

| Runner | Label | Host | Used for |
|---|---|---|---|
| NVIDIA DGX Spark 128GB | `dgx` | `spark-79b7.local` (ARM64, Blackwell GPU) | GPU training and tests |
| MSI WSL2 | `wsl2` | MSI desktop (x86_64) | Triton image build + GKE deploy |

Both training and test containers mount `$HOME/.cache/huggingface` from the DGX host so the `distilbert-base-uncased` tokenizer and IMDB dataset (~200 MB) are downloaded once and reused across runs.

## GitHub Secrets and Variables

| Name | Scope | Type | Description |
|---|---|---|---|
| `WIF_PROVIDER` | org | Secret | Workload Identity Federation provider resource name |
| `GCP_SERVICE_ACCOUNT` | org | Secret | GCP service account email for WIF |
| `REPO_NAME` | repo | Variable | Repository slug used as the K8s namespace |

`MLFLOW_TRACKING_URI` is hardcoded to `http://localhost:5000` in the workflow — not a repo variable. The training container runs with `--network host` so it reaches MLflow on the DGX loopback directly.

## Triton Inference

After deployment, access via port-forward:

```bash
kubectl port-forward -n mlops-torch-triton-gke-pipeline svc/triton 8000:8000

# Health check
curl localhost:8000/v2/health/ready

# Inference (input_ids and attention_mask as INT64 tensors, length 128)
curl -X POST localhost:8000/v2/models/text_classifier/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name": "input_ids",      "shape": [1, 128], "datatype": "INT64", "data": [101, ...]},
      {"name": "attention_mask", "shape": [1, 128], "datatype": "INT64", "data": [1, ...]}
    ]
  }'
```

Logits: index 0 = negative, index 1 = positive. Apply softmax for probabilities.

Triton also exposes gRPC on port 8001 and Prometheus metrics on port 8002.

## Repository Structure

```
.github/workflows/
  ml-train-test.yaml       # Build training image and run pytest (entry point)
  ml-train.yaml            # GPU training on DGX — exports model.onnx as artifact
  ml-deploy.yaml           # Triton image build + GKE deploy on WSL2
ml/
  train.py              # DistilBERT fine-tune + ONNX export + MLflow logging
  test_train.py         # Unit tests for tokenize_batch, evaluate, ONNX export
  requirements.txt      # Local dev dependencies (CPU torch + pytest)
  Dockerfile.train      # GPU training image (nvcr.io/nvidia/pytorch:25.03-py3)
  Dockerfile.serve      # Triton serving image (model.onnx baked in at build time)
  triton_config.pbtxt   # Triton model config (ONNX Runtime backend)
  output/               # Generated at runtime — model.onnx (gitignored)
k8s/
  triton.yaml           # Namespace + Deployment + Service for Triton on GKE
```
