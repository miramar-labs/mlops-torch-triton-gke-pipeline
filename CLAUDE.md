# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Overview

An MLOps pipeline that fine-tunes DistilBERT for text classification (IMDB sentiment), tracks experiments with MLflow, and serves the model on GKE via NVIDIA Triton Inference Server. Training runs on a DGX Station self-hosted GitHub Actions runner with GPU access.

## Repository Structure

```
.github/workflows/
  ml-train-test.yaml    # Build training image and run pytest (entry point)
  ml-train.yaml         # GPU training on DGX — exports model.onnx as GitHub artifact
  ml-deploy.yaml        # Build Triton serving image, push to GAR, deploy to GKE
ml/
  train.py              # DistilBERT fine-tune on IMDB, MLflow logging, ONNX export
  test_train.py         # Unit tests for tokenize_batch, evaluate, ONNX export
  requirements.txt      # Local dev dependencies (CPU torch + pytest)
  Dockerfile.train      # GPU training image (nvcr.io/nvidia/pytorch:25.03-py3)
  Dockerfile.serve      # Triton serving image — model.onnx baked in at build time
  triton_config.pbtxt   # Triton model config: ONNX Runtime backend, input_ids + attention_mask → logits
  output/               # Generated at runtime — model.onnx (gitignored)
k8s/
  triton.yaml           # Namespace mlops-torch-triton-gke-pipeline, Deployment triton, ClusterIP Service
```

## Workflow

Three workflows chain via `workflow_run` (each triggers the next on success):

```
ML Train Test — push to ml/ or workflow_dispatch (runner: dgx, ARM64)
  ├── pytest ml/test_train.py
  └── docker build ml-trainer image

ML Train — triggered by ML Train Test success; or workflow_dispatch (runner: dgx, ARM64, GPU)
  ├── docker build ml-trainer image
  ├── docker run --gpus all --network host → DistilBERT fine-tune + MLflow logging
  ├── export model.onnx via named Docker volume → runner filesystem
  └── upload onnx-model artifact (30-day retention)

ML Deploy — triggered by ML Train success (runner: wsl2, x86_64)
  ├── download onnx-model artifact
  ├── docker build Triton serving image (model.onnx baked in)
  ├── push to GAR (:latest + commit SHA tag)
  └── kubectl apply → GKE, rollout wait 300s
```

**ML Train manual dispatch inputs:**
- `epochs` (default: `3`) — training epochs
- `experiment` (default: `text-classifier`) — MLflow experiment name

**ONNX handoff:** a named Docker volume (`onnx-$RUN_ID`) passes `model.onnx` from the GPU container back to the runner, then `alpine cat` extracts it. The volume is deleted after extraction.

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

MLflow is a persistent service on the DGX host (managed separately). Training containers use `--network host` so they can reach it at `http://localhost:5000`. `MLFLOW_TRACKING_URI` is hardcoded to `http://localhost:5000` in the workflow — not a repo variable.

To view experiment tracking, open the MLflow UI via SSH tunnel:

```bash
ssh -L 5000:localhost:5000 aaron@spark-79b7.local
# then browse http://localhost:5000
```

## GCP / GKE

| Resource | Value |
|---|---|
| Project | `miramar-platform` |
| Cluster | `miramar-shared-gke` (`us-west1-a`) |
| Namespace | `mlops-torch-triton-gke-pipeline` |
| Artifact Registry | `us-west1-docker.pkg.dev/miramar-platform/apps/triton-text-classifier` |
| Auth | Workload Identity Federation — no long-lived keys (pool in project `miramar-cicd`) |

## GitHub Secrets and Variables

| Name | Scope | Type | Description |
|---|---|---|---|
| `WIF_PROVIDER` | org | Secret | WIF provider resource name (see GCP above) |
| `GCP_SERVICE_ACCOUNT` | org | Secret | GCP service account email for WIF |
| `REPO_NAME` | repo | Variable | Repository slug used as the K8s namespace |

## Runners

| Runner | Label | Host | Used for |
|---|---|---|---|
| NVIDIA DGX Spark 128GB | `dgx` | `spark-79b7.local` (ARM64, Blackwell GPU) | GPU training and tests |
| MSI WSL2 | `wsl2` | MSI desktop (x86_64) | Triton image build + GKE deploy |

Runners are managed in [miramar-platform-gcp](https://github.com/miramar-labs-org/miramar-platform-gcp) (`mlabs-runner/` for the Docker image, `scripts/gha/launch-runner.sh` to register). The DGX runner mounts the host Docker socket — `--gpus all` works because the Docker daemon on the DGX host has GPU access. Both containers mount `$HOME/.cache/huggingface` from the DGX host so model weights and the IMDB dataset are downloaded once and reused.

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

## Deploy Notes

- The deploy step strips `kind: Namespace` documents from `k8s/triton.yaml` before applying, so re-deploys don't conflict with an existing namespace.
- The Triton serving image has no `nvidia.com/gpu` resource request in `k8s/triton.yaml` — it runs CPU-only on GKE unless the node pool adds GPU resources separately.
- Image tags: both `:latest` and the commit SHA are pushed; the SHA tag ensures the deployed image always matches the trained model from that exact run.

## Platform Repo

[miramar-platform-gcp](https://github.com/miramar-labs-org/miramar-platform-gcp) at `/home/aaron/git-miramar-labs-org/miramar-platform-gcp` provisions GCP infrastructure (GKE, AR, WIF) and manages the `mlabs-runner` Docker image and launch scripts for the DGX and WSL2 runners. It also hosts the **GKE Cluster Expand** and **GKE Cluster Restore** workflows for temporarily scaling the node pool when testing a Triton deploy.
