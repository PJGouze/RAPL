import json
import numpy as np
import torch

from torch_geometric.loader import DataLoader
from src.dataset.PostProcessedDataset import ProcessedDiskDataset
from src.model.Trainerv3 import Trainer


DATASET = "OntoOmicsKG_step2"

PROCESSED_DIR = (
    f"data_files/{DATASET}/final_biobert_full_v2"
)

CHECKPOINT = (
    "experiments/biobert_full_15ep_seed17/"
    "val_set/model_epoch_4_f1_0.6937.pt"
)

OUTPUT_DIR = (
    "experiments/biobert_full_15ep_seed17/test_set"
)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


# Load test dataset
print("Loading test dataset...")

manifest_path = (
    f"{PROCESSED_DIR}/test/postprocess_manifest.json"
)

with open(manifest_path, "r", encoding="utf-8") as manifest_file:
    test_manifest = json.load(manifest_file)

artifact_metadata = test_manifest.get("artifact_metadata")

if artifact_metadata is None:
    raise RuntimeError(
        "BioBERT schema-v2 test manifest has no artifact_metadata."
    )

print("Test artifact metadata:", artifact_metadata)

test_dataset = ProcessedDiskDataset(
    PROCESSED_DIR,
    split="test",
    artifact_metadata=artifact_metadata,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

print("Test samples:", len(test_dataset))
print("Device:", device)


# Initialize model
trainer = Trainer(
    model_type="GCN",
    bidirectional=False,
    use_stop_mlp=False,
    num_mlp_layers=2,
    num_layers=2,
    num_heads=4,
    K=3,
    in_dims=None,
    emb_size=768,
    hidden_dims=512,
    out_dims=512,
    batch_norm=False,
    dropout=0.2,
    lr=0.001,
    epochs=15,
    device=device,
    dataset_name=DATASET,
    split="test",
    batch_size=1,
    pathTrainAfterEpoch=0,
    wandb_id="biobert_full_15ep_seed17",
)

trainer.to(device)


# Load checkpoint
print("Loading checkpoint:", CHECKPOINT)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True,
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

missing, unexpected = trainer.load_state_dict(
    state_dict,
    strict=False,
)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

if missing or unexpected:
    raise RuntimeError("Checkpoint/model mismatch")


# Evaluate test set
print("Starting test evaluation...")

topic_loss, path_loss, precision, recall = trainer.evaluate(
    test_loader,
    eval_loss=True,
    eval_pr=True,
    is_valid_or_test=True,
    pathtrainingstart=True,
    dataset_idx=0,
    K=50,
    N=1,
    M=50,
)

topic_loss = float(topic_loss)
path_loss = float(path_loss)
precision = float(precision)
recall = float(recall)

total_loss = topic_loss + path_loss

if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0.0


# Print metrics
print()
print("===== TEST METRICS =====")
print("TestTopicLoss:", topic_loss)
print("TestPathLoss :", path_loss)
print("TestTotalLoss:", total_loss)
print("Precision    :", precision)
print("Recall       :", recall)
print("F1           :", f1)


# Save metrics
from pathlib import Path

out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)

metrics = {
    "TestTopicLoss": topic_loss,
    "TestPathLoss": path_loss,
    "TestTotalLoss": total_loss,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
}

with open(out / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

for name, value in metrics.items():
    np.save(out / f"{name}.npy", np.array([value]))

print("Metrics saved to:", OUTPUT_DIR)
