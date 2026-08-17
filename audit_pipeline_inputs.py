import argparse
import pickle
from pathlib import Path

import torch


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def print_sample(sample, index):
    print("\n" + "=" * 70)
    print(f"SAMPLE {index}")
    print("=" * 70)

    if not isinstance(sample, dict):
        print("Type:", type(sample))
        print(sample)
        return

    print("Available keys:", list(sample.keys()))

    for key in (
        "id",
        "question",
        "answer",
        "q_entity",
        "q_entity_id_list",
        "translated_paths",
        "reasoning_paths",
        "paths",
    ):
        if key in sample:
            print(f"\n{key}:")
            print(sample[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="toy")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[3, 4, 5],
    )
    args = parser.parse_args()

    dataset_dir = Path("data_files") / args.name
    processed_dir = dataset_dir / "processed"

    retrieval_path = processed_dir / f"{args.split}_retrieval.pkl"
    raw_path = processed_dir / f"{args.split}.pkl"

    print(f"Dataset: {args.name}")
    print(f"Split: {args.split}")

    for path in (raw_path, retrieval_path):
        print("\n" + "-" * 70)
        print(f"File: {path}")

        if not path.exists():
            print("MISSING")
            continue

        content = load_pickle(path)
        print("Type:", type(content))
        print("Length:", len(content))

        if path == retrieval_path:
            for index in args.indices:
                if 0 <= index < len(content):
                    print_sample(content[index], index)
                else:
                    print(f"\nIndex {index}: OUT OF RANGE")

    annotation_dir = (
        dataset_dir / "annotated_paths_LLM" / args.split
    )
    annotations = sorted(annotation_dir.glob("sample_*.txt"))

    pyg_dir = processed_dir / args.split
    pyg_files = sorted(pyg_dir.glob("data_*.pt"))

    print("\n" + "-" * 70)
    print(f"LLM annotations: {len(annotations)}")
    print(f"PyG files: {len(pyg_files)}")

    embedding_files = list(
        (dataset_dir / "emb").rglob(f"{args.split}.pth")
    )

    for embedding_path in embedding_files:
        embeddings = torch.load(
            embedding_path,
            map_location="cpu",
            weights_only=False,
        )

        try:
            length = len(embeddings)
        except TypeError:
            length = "not available"

        print(f"Embedding file: {embedding_path}")
        print(f"Embedding object type: {type(embeddings)}")
        print(f"Embedding length: {length}")


if __name__ == "__main__":
    main()
