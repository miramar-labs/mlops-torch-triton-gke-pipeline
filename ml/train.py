import os
import numpy as np

import dill._dill
if not hasattr(dill._dill, "PY3"):
    dill._dill.PY3 = True
if not hasattr(np, "object"):
    np.object = object  # removed in NumPy 1.24; older datasets versions still reference it

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

EPOCHS = int(os.environ.get("EPOCHS", 3))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
MAX_LEN = 128
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "/output"

def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            preds = model(input_ids=ids, attention_mask=mask).logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total


def main():
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "text-classifier"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    dataset = load_dataset("imdb")

    cols = ["input_ids", "attention_mask", "label"]
    train_ds = dataset["train"].map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    train_ds.set_format("torch", columns=cols)
    test_ds = dataset["test"].map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    test_ds.set_format("torch", columns=cols)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "model": MODEL_NAME,
            "dataset": "imdb",
        })

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_acc = evaluate(model, test_loader, device)
            mlflow.log_metrics({"train_loss": avg_loss, "val_accuracy": val_acc}, step=epoch)
            print(f"Epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        # Export to ONNX
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.eval()
        dummy_ids = torch.ones(1, MAX_LEN, dtype=torch.long, device=device)
        dummy_mask = torch.ones(1, MAX_LEN, dtype=torch.long, device=device)
        onnx_path = os.path.join(OUTPUT_DIR, "model.onnx")

        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
        )

        mlflow.log_artifact(onnx_path)
        print(f"ONNX model saved to {onnx_path}")


if __name__ == "__main__":
    main()

