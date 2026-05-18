# mlops-torch-triton-gke-pipeline

GPU ML training pipeline: fine-tune DistilBERT for text classification on a DGX Station, track experiments with MLflow, serve the model on GKE via Triton Inference Server.

[![ML Train Test](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-train-test.yaml/badge.svg)](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-train-test.yaml)
[![ML Train](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-train.yaml/badge.svg)](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-train.yaml)
[![ML Deploy](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-deploy.yaml/badge.svg)](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions/workflows/ml-deploy.yaml)

## Links

- **[GCP Artifact Registry](https://console.cloud.google.com/artifacts/docker/miramar-platform/us-west1/apps?project=miramar-platform)** — `us-west1-docker.pkg.dev/miramar-platform/apps/triton-text-classifier`
- **[GKE Workloads](https://console.cloud.google.com/kubernetes/workload/overview?project=miramar-platform)** — `triton` deployment in namespace `mlops-torch-triton-gke-pipeline` on `miramar-shared-gke`
- **[GitHub Actions](https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline/actions)** — workflow run history

## Local Development

```bash
# Create and activate a pyenv virtualenv for this workspace
pyenv virtualenv 3.12.2 mlops-torch-triton
pyenv local mlops-torch-triton

# Install dependencies (CPU torch — enough for tests and linting)
pip install -r ml/requirements.txt
```

VS Code will use the `mlops-torch-triton` interpreter automatically via `.python-version`. Tests in `ml/test_train.py` show as **skipped** locally (no GPU/CUDA torch) and run fully in the training container on the DGX.

## Pipeline

```
ML Train Test — workflow_dispatch or push to ml/ (dgx, ARM64)
  ├── docker build → ml-trainer image
  └── pytest → test_train.py

ML Train — triggered by ML Train Test success (dgx, ARM64, GPU)
  ├── docker build → ml-trainer image
  ├── docker run --gpus all → DistilBERT fine-tune on IMDB
  ├── log metrics → MLflow (host.docker.internal:5000)
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
| **ML Train Test** | `ml-train-test.yaml` | `dgx` | Push to `ml/` or manual |
| **ML Train** | `ml-train.yaml` | `dgx` | Auto on ML Train Test success; or manual |
| **ML Deploy** | `ml-deploy.yaml` | `wsl2` | Auto on ML Train success |

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

## MLflow

MLflow runs on the DGX host and is accessible to training containers via `host.docker.internal:5000`.

| Detail | Value |
|---|---|
| DGX folder | `/home/aaron/mlflow` |
| Python env | `pyMlFlow` virtualenv (pyenv) |
| Port | `5000` |
| tmux session | `mlflow` |

**Start the server on the DGX:**
```bash
cd /home/aaron/mlflow
pyenv activate pyMlFlow
tmux new -s mlflow
python -m mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
# Ctrl+B then D to detach
```

**Reattach to check logs:**
```bash
tmux attach -t mlflow
```

**Access the UI from your laptop** (SSH tunnel):
```powershell
ssh -L 5000:localhost:5000 aaron@spark-79b7.local
```
Then open **http://localhost:5000** in your browser.

## GCP / GKE

| Resource | Value |
|---|---|
| Project | `miramar-platform` |
| Cluster | `miramar-shared-gke` (`us-west1-a`) |
| Namespace | `mlops-torch-triton-gke-pipeline` |
| Artifact Registry | `us-west1-docker.pkg.dev/miramar-platform/apps/triton-text-classifier` |
| Auth | Workload Identity Federation — no long-lived keys |

## Runners

Two self-hosted runners are required. The runner image and launch scripts live in [github-actions-hello](https://github.com/miramar-labs-org/github-actions-hello).

| Runner | Label | Host | Used for |
|---|---|---|---|
| DGX Station | `dgx` | `spark-79b7.local` (ARM64) | GPU training |
| MSI WSL2 | `wsl2` | MSI desktop (x86_64) | Triton image build + GKE deploy |

## GitHub Secrets and Variables

| Name | Type | Description |
|---|---|---|
| `WIF_PROVIDER` | Secret | Workload Identity Federation provider resource name |
| `GCP_SERVICE_ACCOUNT` | Secret | GCP service account email for WIF |
| `MLFLOW_TRACKING_URI` | Variable | `http://host.docker.internal:5000` |
| `REPO_NAME` | Variable | Repository slug used as the K8s namespace |

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
  ml-train-test.yaml    # Build training image and run pytest (entry point)
  ml-train.yaml         # GPU training on DGX — exports model.onnx as artifact
  ml-deploy.yaml        # Triton image build + GKE deploy on WSL2
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
