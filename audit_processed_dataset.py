import argparse
from pathlib import Path

import torch


def natural_key(path: Path):
    """Sort data_2.pt before data_10.pt."""
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return path.name


def get_tensor(data, *attribute_names):
    """Return the first available tensor among several possible names."""
    for attribute_name in attribute_names:
        if hasattr(data, attribute_name):
            value = getattr(data, attribute_name)

            if value is not None:
                return torch.as_tensor(value).detach().cpu()

    return None


def describe(name, tensor):
    """Create a compact description of a label tensor."""
    if tensor is None:
        return f"{name}: MISSING"

    unique_values = torch.unique(tensor).tolist()

    if len(unique_values) > 20:
        unique_values = unique_values[:20] + ["..."]

    return (
        f"{name}: "
        f"shape={tuple(tensor.shape)}, "
        f"sum={tensor.sum().item():.0f}, "
        f"unique={unique_values}"
    )


def load_pyg_file(file_path: Path):
    """Load one trusted PyTorch Geometric file on CPU."""
    return torch.load(
        file_path,
        map_location="cpu",
        weights_only=False,
    )


def audit_split(base_dir: Path, split: str, displayed_samples: int):
    """Audit all processed PyG files from one dataset split."""
    split_dir = base_dir / split
    files = sorted(split_dir.glob("data_*.pt"), key=natural_key)

    print("\n" + "=" * 70)
    print(f"SPLIT: {split}")
    print(f"Directory: {split_dir}")
    print(f"Number of files: {len(files)}")
    print("=" * 70)

    if not files:
        print("ERROR: no data_*.pt file found")
        return

    total_nodes = 0
    candidate_positive_total = 0
    topic_positive_total = 0
    llm_positive_total = 0
    valid_path_entries_total = 0

    samples_with_candidates = 0
    samples_with_topic_labels = 0
    samples_with_llm_labels = 0
    samples_with_valid_paths = 0

    labels_outside_candidates = 0
    incompatible_label_shapes = 0
    loading_errors = 0

    for file_index, file_path in enumerate(files):
        try:
            data = load_pyg_file(file_path)
        except Exception as error:
            loading_errors += 1
            print(f"\nERROR while loading {file_path.name}: {error}")
            continue

        candidates = get_tensor(
            data,
            "topic_candidates",
            "topic_candidate",
        )

        topic_labels = get_tensor(
            data,
            "topic_labels",
            "topic_label",
        )

        llm_labels = get_tensor(
            data,
            "llm_topic_labels",
            "llm_topic_label",
        )

        path_labels = get_tensor(
            data,
            "path_label",
            "path_labels",
        )

        if hasattr(data, "x") and data.x is not None:
            total_nodes += int(data.x.shape[0])

        if candidates is not None:
            positive_mask = candidates > 0
            candidate_positive_total += int(positive_mask.sum())
            samples_with_candidates += int(positive_mask.any())

        if topic_labels is not None:
            positive_mask = topic_labels > 0
            topic_positive_total += int(positive_mask.sum())
            samples_with_topic_labels += int(positive_mask.any())

        if llm_labels is not None:
            positive_mask = llm_labels > 0
            llm_positive_total += int(positive_mask.sum())
            samples_with_llm_labels += int(positive_mask.any())

        if candidates is not None and topic_labels is not None:
            if candidates.shape == topic_labels.shape:
                labels_outside_candidates += int(
                    ((topic_labels > 0) & (candidates <= 0)).sum()
                )
            else:
                incompatible_label_shapes += 1

        if path_labels is not None:
            valid_mask = path_labels != -1
            valid_path_entries_total += int(valid_mask.sum())
            samples_with_valid_paths += int(valid_mask.any())

        if file_index < displayed_samples:
            x_shape = (
                tuple(data.x.shape)
                if hasattr(data, "x") and data.x is not None
                else "MISSING"
            )

            print(f"\nFile: {file_path.name}")
            print(f"x shape: {x_shape}")
            print(describe("topic_candidates", candidates))
            print(describe("topic_labels", topic_labels))
            print(describe("llm_topic_labels", llm_labels))
            print(describe("path_label", path_labels))

    print("\n--- SPLIT SUMMARY ---")
    print(f"Total graph nodes                  : {total_nodes}")
    print(f"Positive topic candidates          : {candidate_positive_total}")
    print(f"Positive standard topic labels     : {topic_positive_total}")
    print(f"Positive LLM topic labels          : {llm_positive_total}")
    print(f"Valid path-label entries (!= -1)   : {valid_path_entries_total}")
    print(
        "Samples with topic candidates      : "
        f"{samples_with_candidates}/{len(files)}"
    )
    print(
        "Samples with standard topic labels : "
        f"{samples_with_topic_labels}/{len(files)}"
    )
    print(
        "Samples with LLM topic labels      : "
        f"{samples_with_llm_labels}/{len(files)}"
    )
    print(
        "Samples with valid path labels     : "
        f"{samples_with_valid_paths}/{len(files)}"
    )
    print(f"Topic labels outside candidates    : {labels_outside_candidates}")
    print(f"Incompatible candidate/label shapes: {incompatible_label_shapes}")
    print(f"File loading errors                : {loading_errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit labels stored in a processed PyG dataset."
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name, for example: toy",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Splits to inspect.",
    )

    parser.add_argument(
        "--displayed_samples",
        type=int,
        default=3,
        help="Number of detailed samples displayed per split.",
    )

    args = parser.parse_args()

    base_dir = Path("data_files") / args.name / "processed"

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Processed dataset directory not found: {base_dir}"
        )

    print(f"Auditing dataset: {args.name}")
    print(f"Base directory: {base_dir.resolve()}")

    for split in args.splits:
        audit_split(
            base_dir=base_dir,
            split=split,
            displayed_samples=args.displayed_samples,
        )


if __name__ == "__main__":
    main()
