# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Overview

An MLOps pipeline that fine-tunes DistilBERT for text classification (IMDB sentiment), tracks experiments with MLflow, and serves the model on GKE via NVIDIA Triton Inference Server. Training runs on a DGX Station self-hosted GitHub Actions runner with GPU access.

## Repository Structure

```
.github/workflows/
  ml-train.yaml         # Main workflow: train → export ONNX → build Triton image → deploy to GKE
ml/
  train.py              # DistilBERT fine-tune on IMDB, MLflow logging, ONNX export
  Dockerfile.train      # GPU training image (pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime)
  Dockerfile.serve      # Triton serving image — model.onnx baked in at build time
  triton_config.pbtxt   # Triton model config: ONNX Runtime backend, input_ids + attention_mask → logits
  output/               # Generated at runtime — model.onnx (gitignored)
k8s/
  triton.yaml           # Namespace mlops-torch-triton-gke-pipeline, Deployment triton, ClusterIP Service
```

## Workflow

**File:** `.github/workflows/ml-train.yaml`  
**Trigger:** `workflow_dispatch` only (manual)  
**Runner:** `dgx-spark` (DGX Station, ARM64, NVIDIA GPU) via `vars.RUNNER_LABELS`

**Jobs:**
1. `train` — builds training image, runs `docker run --gpus all` with MLflow + ONNX export, builds Triton serving image, pushes to GAR
2. `deploy-triton` (needs: train) — WIF auth, GKE credentials, `kubectl apply k8s/triton.yaml`, waits for rollout

**Inputs:**
- `epochs` (default: `3`) — training epochs
- `experiment` (default: `text-classifier`) — MLflow experiment name

## Model

| Property | Value |
|---|---|
| Base model | `distilbert-base-uncased` (HuggingFace) |
| Task | Binary sentiment classification |
| Dataset | `imdb` from HuggingFace `datasets` — auto-downloaded |
| Sequence length | 128 tokens (pad/truncate) |
| ONNX inputs | `input_ids` (INT64, [batch, 128]), `attention_mask` (INT64, [batch, 128]) |
| ONNX output | `logits` (FP32, [batch, 2]) |

## MLflow

Self-hosted on the DGX host. Start it before triggering the workflow:

```bash
pip install mlflow
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

Training containers reach it via `host.docker.internal:5000` (Linux Docker bridge). The `MLFLOW_TRACKING_URI` repo variable is already set to `http://host.docker.internal:5000`.

## GCP / GKE

| Resource | Value |
|---|---|
| Project | `miramar-platform` |
| Cluster | `miramar-shared-gke` (`us-west1-a`) |
| Namespace | `mlops-torch-triton-gke-pipeline` |
| Artifact Registry | `us-west1-docker.pkg.dev/miramar-platform/apps/triton-text-classifier` |
| WIF Pool | `projects/423801268174/locations/global/workloadIdentityPools/github-actions/providers/github` (project: `miramar-cicd`) |
| Auth | Workload Identity Federation — no long-lived keys |

## GitHub Secrets and Variables

| Name | Type | Value / Notes |
|---|---|---|
| `WIF_PROVIDER` | Secret | WIF provider resource name (see GCP above) |
| `GCP_SERVICE_ACCOUNT` | Secret | GCP service account email |
| `MLFLOW_TRACKING_URI` | Variable | `http://host.docker.internal:5000` |
| `RUNNER_LABELS` | Variable | `dgx-spark` |

## Runner

The DGX runner must be:
1. Registered for this repo with label `dgx-spark`
2. Running as a Docker container using `ghcr.io/miramar-labs-org/github-runner-mlops-torch-triton-gke-pipeline:latest`
3. Started via `./runner/launch.sh TOKEN` from the [github-actions-hello](https://github.com/miramar-labs-org/github-actions-hello) repo

The runner container mounts the host Docker socket — `--gpus all` in the training step works because the Docker daemon on the DGX host has GPU access.

## Triton Inference

After deployment, test via port-forward:

```bash
kubectl port-forward -n mlops-torch-triton-gke-pipeline svc/triton 8000:8000

# Health
curl localhost:8000/v2/health/ready

# Inference (tokenize your text to input_ids + attention_mask, pad to length 128)
curl -X POST localhost:8000/v2/models/text_classifier/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name": "input_ids",      "shape": [1, 128], "datatype": "INT64", "data": [101, ...]},
      {"name": "attention_mask", "shape": [1, 128], "datatype": "INT64", "data": [1, ...]}
    ]
  }'
```

Logits output: index 0 = negative, index 1 = positive. Apply softmax for probabilities.

## Sibling Repo

[github-actions-hello](https://github.com/miramar-labs-org/github-actions-hello) at `/home/aaron/git-miramar-labs-org/github-actions-hello` contains the runner image (`ghcr.io/miramar-labs-org/github-runner-mlops-torch-triton-gke-pipeline`), runner launch scripts, and the textlyze app pipeline. This repo was created from that project.
