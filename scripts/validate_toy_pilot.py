"""Validate an isolated toy training pilot without changing dataset artifacts."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.dataset.PostProcessedDataset import ProcessedDiskDataset
from src.model.Trainerv3 import Trainer


def _require_finite_numbers(value, context):
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite value at {context}: {value!r}.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_numbers(item, f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_numbers(item, f"{context}.{key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, default=5)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    config = json.loads((output_dir / "run_config.json").read_text())
    history = json.loads((output_dir / "training_history.json").read_text())
    parameter_audit = json.loads(
        (output_dir / "parameter_audit.json").read_text()
    )
    _require_finite_numbers(history, "training_history")

    epochs = history["epochs"]
    if len(epochs) != args.expected_epochs:
        raise RuntimeError(
            f"Expected {args.expected_epochs} epochs, found {len(epochs)}."
        )
    if history["optimizer_step_count"] <= 0:
        raise RuntimeError("The optimizer did not perform an update.")
    for component, audit in parameter_audit.items():
        if not audit["changed"]:
            raise RuntimeError(f"Selected {component} parameter did not change.")
    if any(
        item["parameters_with_nonfinite_gradients"]
        for item in history["gradient_audit"]
    ):
        raise RuntimeError("A non-finite gradient was recorded.")
    if not any(
        item["parameters_with_nonzero_gradients"] > 0
        for item in history["gradient_audit"]
    ):
        raise RuntimeError("No non-zero gradient was recorded.")

    for epoch in epochs:
        if epoch["train_path_decision_count"] <= 0:
            raise RuntimeError(
                f"Epoch {epoch['epoch']} has no supervised training path decision."
            )
        if epoch["validation_path_decision_count"] <= 0:
            raise RuntimeError(
                f"Epoch {epoch['epoch']} has no validation path decision."
            )
        for prefix in ("train", "validation"):
            if epoch[f"{prefix}_loss_dtypes"] != ["float32"]:
                raise RuntimeError(
                    f"Expected float32 {prefix} path cross-entropy, found "
                    f"{epoch[f'{prefix}_loss_dtypes']!r}."
                )

    checkpoint_paths = sorted(output_dir.glob("epoch_*_trainLoss_*.pt"))
    if len(checkpoint_paths) != args.expected_epochs:
        raise RuntimeError(
            f"Expected {args.expected_epochs} epoch checkpoints, found "
            f"{len(checkpoint_paths)}."
        )
    payloads = [
        (path, torch.load(path, map_location="cpu", weights_only=False))
        for path in checkpoint_paths
    ]
    checkpoint_path, checkpoint = max(payloads, key=lambda item: item[1]["epoch"])
    expected_keys = {
        "epoch",
        "metrics",
        "model_state_dict",
        "optimizer_state_dict",
    }
    if set(checkpoint) != expected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {sorted(checkpoint)}."
        )
    if checkpoint["epoch"] != args.expected_epochs:
        raise RuntimeError(
            f"Latest checkpoint epoch is {checkpoint['epoch']}, expected "
            f"{args.expected_epochs}."
        )

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config["device"] = 0 if torch.cuda.is_available() else "cpu"
    trainer = Trainer(**config)
    trainer.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if any(not torch.isfinite(parameter).all() for parameter in trainer.parameters()):
        raise RuntimeError("A reloaded model parameter is non-finite.")

    validation_dataset = ProcessedDiskDataset("data_files/toy/processed", "val")
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    val_topic, val_path, precision, recall = trainer.evaluate(
        validation_loader,
        eval_loss=True,
        eval_pr=True,
        pathtrainingstart=True,
        dataset_idx=1,
    )
    reload_values = [float(val_topic), val_path, precision, recall]
    if not all(value is None or math.isfinite(float(value)) for value in reload_values):
        raise RuntimeError(f"Non-finite post-reload validation: {reload_values!r}.")
    if trainer.validation_grad_enabled_observations[-1]:
        raise RuntimeError("Post-reload validation did not run under torch.no_grad().")

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_keys": sorted(checkpoint),
        "model_state_loaded": True,
        "optimizer_state_loaded": True,
        "device": str(trainer.device),
        "validation_topic_loss": float(val_topic),
        "validation_path_loss": val_path,
        "precision": precision,
        "recall": recall,
        "validation_grad_enabled": trainer.validation_grad_enabled_observations[-1],
        "validation_path_statistics": trainer.last_evaluation_path_statistics,
    }
    (output_dir / "pilot_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
