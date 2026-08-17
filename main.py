import os
# Must be set *before* importing torch or allocating GPU tensors:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import hashlib
import json
import random
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import numpy as np
import sys
import pandas as pd
import time
import torch
import torch.nn.functional as F

# from src.model.Trainer import Trainer
from src.model.Trainerv3 import Trainer
from src.dataset.PostProcessedDataset import ProcessedDiskDataset
from src.dataset.utils import load_pickles,decode_path
from termcolor import colored
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def color_print(text, color='green'):
    print(colored(text, color))

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a GNN-based model with path prediction.")

    # Add arguments for Trainer configuration
    parser.add_argument("--model_type", type=str, default="GCN", help="Type of GNN model: GCN, GAT, SGConv, GIN.")
    parser.add_argument("--bidirectional", action="store_true",default=False, help="Use bidirectional GNN if set.")
    parser.add_argument("--use_stop_mlp", action="store_true",default=False, help="Use a separate MLP for stop token if set.")
    parser.add_argument("--num_mlp_layers", type=int, default=2, help="Number of MLP layers.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads in GAT.")
    parser.add_argument("--K", type=int, default=3, help="K for SGConv.")
    parser.add_argument(
        "--in_dims",
        type=int,
        default=None,
        help="Input node feature dimension. If omitted, inferred as 3 * emb_size.",
    )
    parser.add_argument(
        "--emb_size",
        type=int,
        default=None,
        help=(
            "Text embedding dimension. "
            "For schema-v2 datasets it is inferred from the manifest. "
            "Legacy schema-v1 datasets default to 384."
        ),
    )
    parser.add_argument("--hidden_dims", type=int, default=512, help="Hidden dimension size in GNN.")
    parser.add_argument("--out_dims", type=int, default=512, help="Output dimension for GNN.")
    parser.add_argument("--batch_norm", action="store_true", help="Use batch normalization if set.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout ratio.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--device", type=int, default=0, help="GPU device index. Use -1 for CPU.")
    
    # Add dataset-related arguments
    parser.add_argument("--dataset_name", type=str, default="toy", help="Name or path of the dataset.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split: train, val, test.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the DataLoader.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic random seed.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Isolated directory for checkpoints and metric files.",
    )

    parser.add_argument(
        "--final-dataset-dir",
        dest="final_dataset_dir",
        type=str,
        default=None,
        help=(
            "Explicit processed dataset root containing train/val/test. "
            "If omitted, use data_files/<dataset_name>/final_filtered."
        ),
    )
    
    parser.add_argument("--pathTrainAfterEpoch", type=int, default=0, help="start training path loss after #epoch")
    parser.add_argument("--wandb_id", type=str, default='1')

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Use only the first N training samples for a benchmark run.",
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Use only the first N validation samples for a benchmark run.",
    )

    return parser


def _parameter_hashes(trainer):
    selected = {
        "gnn": "convs.0.lin.weight",
        "topic": "r_mlp.0.weight",
        "path_stop": "stop_emb",
    }
    parameters = dict(trainer.named_parameters())
    return {
        component: hashlib.sha256(
            parameters[name].detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for component, name in selected.items()
    }


def _validate_final_dataset_manifests(final_dataset_dir, args):
    """Validate train/val dataset manifests before training starts."""

    manifests = {}

    for split in ("train", "val"):
        split_dir = os.path.join(final_dataset_dir, split)

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Final dataset directory {final_dataset_dir!r} is missing "
                f"required split directory {split!r}."
            )

        manifest_path = os.path.join(
            split_dir,
            "postprocess_manifest.json",
        )

        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"Missing postprocess manifest for split={split!r}: "
                f"{manifest_path}"
            )

        with open(manifest_path, encoding="utf-8") as f:
            manifests[split] = json.load(f)

    train_manifest = manifests["train"]
    val_manifest = manifests["val"]

    train_schema = train_manifest.get("schema_version")
    val_schema = val_manifest.get("schema_version")

    if train_schema != val_schema:
        raise ValueError(
            "Train/val manifest schema mismatch: "
            f"train={train_schema}, val={val_schema}."
        )

    if train_schema not in (1, 2):
        raise ValueError(
            f"Unsupported postprocess manifest schema: {train_schema!r}."
        )

    # Legacy baseline: preserve current behavior.
    if train_schema == 1:
        if args.emb_size is None:
            args.emb_size = 384

        print("Dataset manifest mode     : legacy schema v1")
        print("Embedding dimension       :", args.emb_size)
        return manifests

    train_metadata = train_manifest.get("artifact_metadata")
    val_metadata = val_manifest.get("artifact_metadata")

    if not isinstance(train_metadata, dict):
        raise ValueError(
            "Train schema-v2 manifest is missing artifact_metadata."
        )

    if not isinstance(val_metadata, dict):
        raise ValueError(
            "Val schema-v2 manifest is missing artifact_metadata."
        )

    compatibility_fields = (
        "dataset_name",
        "embedding_model",
        "embedding_dim",
        "llm_name",
        "graph_processed_dir",
    )

    for field in compatibility_fields:
        train_value = train_metadata.get(field)
        val_value = val_metadata.get(field)

        if train_value != val_value:
            raise ValueError(
                f"Train/val artifact mismatch for {field!r}: "
                f"train={train_value!r}, val={val_value!r}."
            )

    manifest_dataset = train_metadata.get("dataset_name")

    if manifest_dataset != args.dataset_name:
        raise ValueError(
            "Dataset name mismatch: "
            f"CLI={args.dataset_name!r}, "
            f"manifest={manifest_dataset!r}."
        )

    embedding_dim = train_metadata.get("embedding_dim")

    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError(
            f"Invalid embedding_dim in manifest: {embedding_dim!r}."
        )

    if args.emb_size is None:
        args.emb_size = embedding_dim
    elif args.emb_size != embedding_dim:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"--emb_size={args.emb_size}, "
            f"dataset manifest embedding_dim={embedding_dim}. "
            "Use the embedding dimension recorded in the final dataset manifest."
        )

    print("Dataset manifest mode     : experimental schema v2")
    print("Embedding model           :", train_metadata["embedding_model"])
    print("Embedding dimension       :", embedding_dim)
    print("Annotation LLM            :", train_metadata["llm_name"])

    return manifests


def main():
    args = build_arg_parser().parse_args()

    dataset = args.dataset_name
    final_processed_dir = (
        args.final_dataset_dir
        if args.final_dataset_dir is not None
        else f"data_files/{dataset}/final_filtered"
    )

    dataset_manifests = _validate_final_dataset_manifests(
        final_processed_dir,
        args,
    )

    train_artifact_metadata = (
        dataset_manifests["train"].get("artifact_metadata")
        if dataset_manifests["train"].get("schema_version") == 2
        else None
    )

    val_artifact_metadata = (
        dataset_manifests["val"].get("artifact_metadata")
        if dataset_manifests["val"].get("schema_version") == 2
        else None
    )

    print(f"Final dataset directory   : {final_processed_dir}")

    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # -------------------------
    # 0. init wandb
    # -------------------------

    # Start a new wandb run to track this script.
    # run = wandb.init(
    #     # Set the wandb project where this run will be logged.
    #     project=f"KGQA_{args.dataset_name}_{args.model_type}_wandb_{args.wandb_id}",
    #     # Track hyperparameters and run metadata.
    #     config={
    #         "bidirectional": args.bidirectional,
    #         "num_layers": args.num_layers,
    #         "use_stop_mlp": args.use_stop_mlp,
    #         "epochs": 100,
    #         "num_heads": args.num_heads,
    #         "K": args.K
    #     },
    # )
    # args.wandb = run
    save_dir = args.output_dir or f'experiments/{args.dataset_name}/saved_models/{args.wandb_id}/{args.model_type}_bidirectional_{args.bidirectional}_numLayers_{args.num_layers}_useStopMlp_{args.use_stop_mlp}_numHeads_{args.num_heads}_K_{args.K}/seed_{seed}/'
    if args.output_dir is not None and os.path.exists(save_dir):
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {save_dir}"
        )
    os.makedirs(save_dir, exist_ok=args.output_dir is None)
    with open(os.path.join(save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
        f.write("\n")
    
    
    color_print(args,'cyan')
    
    # -------------------------
    # 1. load datasets
    # -------------------------  
    train_set1 = ProcessedDiskDataset(
        processed_dir=final_processed_dir,
        split="train",
        artifact_metadata=train_artifact_metadata,
    )

    val_set1 = ProcessedDiskDataset(
        processed_dir=final_processed_dir,
        split="val",
        artifact_metadata=val_artifact_metadata,
    )

    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive.")

        train_count = min(
            args.max_train_samples,
            len(train_set1),
        )

        train_set1 = Subset(
            train_set1,
            range(train_count),
        )

    if args.max_val_samples is not None:
        if args.max_val_samples <= 0:
            raise ValueError("--max-val-samples must be positive.")

        val_count = min(
            args.max_val_samples,
            len(val_set1),
        )

        val_set1 = Subset(
            val_set1,
            range(val_count),
        )

    print(
        f"Training samples used      : {len(train_set1)}"
    )
    print(
        f"Validation samples used    : {len(val_set1)}"
    )

    
    # -------------------------
    # 2. Create DataLoader
    # -------------------------
    generator = torch.Generator().manual_seed(seed)
    train_dataloader1 = DataLoader(
        train_set1,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_dataloader1 = DataLoader(
        val_set1,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    color_print(f"Created DataLoader with batch_size={args.batch_size}.",'red')
    
    # -------------------------
    # 3. Initialize Trainer
    # -------------------------

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    color_print (f"use_device:{device}",'red')
    args.device = device
    args = vars(args)
    trainer = Trainer(**args)
    initial_parameter_hashes = _parameter_hashes(trainer)

    color_print("Trainer initialized.",'red')

    # -------------------------
    # 4. Train the Model
    # -------------------------
    trainer.to(device)
    print ('formal training with LLM labels')
    #trainer.fit_mixture(train_dataloader1,val_dataloader1,save_dir=save_dir,pathtrainingstart=True)
    history = trainer.fit(
                train_dataloader1,
                val_dataloader1,
                save_dir=save_dir,
                pathtrainingstart=True
            )
    final_parameter_hashes = _parameter_hashes(trainer)
    parameter_audit = {
        component: {
            "before": initial_parameter_hashes[component],
            "after": final_parameter_hashes[component],
            "changed": initial_parameter_hashes[component]
            != final_parameter_hashes[component],
        }
        for component in initial_parameter_hashes
    }
    with open(os.path.join(save_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(save_dir, "parameter_audit.json"), "w", encoding="utf-8") as f:
        json.dump(parameter_audit, f, indent=2, sort_keys=True)
        f.write("\n")
if __name__ == "__main__":
    main()
