"""Build the final graph dataset from retrieval and annotation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Callable, Sequence, TypeAlias

RelationPath: TypeAlias = tuple[str, ...]
AnnotatedTextLabels: TypeAlias = list[list[RelationPath]]

_SAMPLE_FILE_PATTERN = re.compile(r"^sample_(\d+)\.txt$")
_NUMBERED_LINE_PATTERN = re.compile(r"^\s*\d+\s*[.):\-]\s*(.*?)\s*$")
_SUPPORTED_SPLITS = {"train", "val", "test"}
_SPLIT_ALIASES = {"valid": "val", "validation": "val"}


class AnnotationParseError(ValueError):
    """Raised when an annotation contains a malformed path."""


def _normalize_relation_item(item: object) -> str:
    """Normalize an LLM relation token without modifying source annotations."""
    normalized = str(item).strip().strip("'\"").strip()
    if "->" in normalized:
        normalized = normalized.split("->", 1)[0].strip()
    return normalized


def normalize_split(split: str) -> str:
    """Return the canonical dataset split name."""
    canonical_split = _SPLIT_ALIASES.get(split, split)
    if canonical_split not in _SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported split {split!r}; expected train, val/valid/validation, or test."
        )
    return canonical_split


def _parse_rational_paths(
    file_dir: str | os.PathLike[str],
    file_name: str,
    *,
    split: str,
    sample_index: int,
    allow_empty: bool = False,
) -> list[RelationPath]:
    """Parse old relation-list and new entity/relation arrow annotations."""
    full_path = Path(file_dir) / file_name
    parsed_paths: list[RelationPath] = []

    with full_path.open("r", encoding="utf-8") as annotation_file:
        for line_number, raw_line in enumerate(annotation_file, start=1):
            numbered_match = _NUMBERED_LINE_PATTERN.match(raw_line.strip())
            if numbered_match is None:
                continue

            path_text = numbered_match.group(1).strip().strip("*").strip()
            if path_text.startswith("[") and path_text.endswith("]"):
                relations = tuple(
                    _normalize_relation_item(relation)
                    for relation in path_text[1:-1].split(",")
                )
                if not relations or any(not relation for relation in relations):
                    raise AnnotationParseError(
                        f"Malformed annotation for split={split!r}, file={full_path}, "
                        f"sample_index={sample_index}, line={line_number}: "
                        "relation-list paths must contain non-empty relation names."
                    )
                parsed_paths.append(relations)
                continue

            if "->" not in path_text:
                continue

            elements = tuple(part.strip().strip("*") for part in path_text.split("->"))
            if len(elements) < 3 or len(elements) % 2 == 0 or any(not part for part in elements):
                raise AnnotationParseError(
                    f"Malformed annotation for split={split!r}, file={full_path}, "
                    f"sample_index={sample_index}, line={line_number}: arrow paths must "
                    "alternate entity -> relation -> entity and end with an entity."
                )
            parsed_paths.append(tuple(elements[1::2]))

    if not parsed_paths and allow_empty:
        return []
    if not parsed_paths:
        raise ValueError(
            f"Empty annotation for split={split!r}, file={full_path}, "
            f"sample_index={sample_index}: no valid rational path was extracted."
        )
    return parsed_paths


def _collect_annotated_paths(
    folder_path: str | os.PathLike[str],
    *,
    split: str,
    expected_samples: int,
    allow_extra_samples: bool = False,
    allow_invalid: bool = False,
    diagnostics: list[dict] | None = None,
) -> AnnotatedTextLabels:
    """Load annotations by explicit numeric sample index without positional shifting."""
    canonical_split = normalize_split(split)
    resolved_path = Path(folder_path)
    if not resolved_path.is_dir():
        raise FileNotFoundError(
            f"Annotation directory missing for split={canonical_split!r}: {resolved_path}"
        )

    indexed_files: dict[int, str] = {}
    for file_name in os.listdir(resolved_path):
        match = _SAMPLE_FILE_PATTERN.fullmatch(file_name)
        if match is None:
            continue
        sample_index = int(match.group(1))
        if sample_index in indexed_files:
            raise ValueError(
                f"Duplicate annotation index for split={canonical_split!r}, "
                f"path={resolved_path}, sample_index={sample_index}."
            )
        indexed_files[sample_index] = file_name

    expected_indices = set(range(expected_samples))
    actual_indices = set(indexed_files)
    missing_indices = sorted(expected_indices - actual_indices)
    unexpected_indices = sorted(actual_indices - expected_indices)
    if missing_indices or (unexpected_indices and not allow_extra_samples):
        raise ValueError(
            f"Annotation sample index mismatch for split={canonical_split!r}, "
            f"path={resolved_path}: missing={missing_indices}, "
            f"unexpected={unexpected_indices}, expected_samples={expected_samples}, "
            f"allow_extra_samples={allow_extra_samples}."
        )

    labels = []
    for sample_index in range(expected_samples):
        try:
            labels.append(_parse_rational_paths(
                resolved_path, indexed_files[sample_index],
                split=canonical_split, sample_index=sample_index,
                allow_empty=allow_invalid,
            ))
        except (ValueError, AnnotationParseError) as error:
            if not allow_invalid:
                raise
            labels.append([])
            if diagnostics is not None:
                diagnostics.append({
                    "original_index": sample_index,
                    "reason": "invalid_llm_annotation",
                    "error": str(error),
                    "llm_annotations": [],
                })
    return labels


def filter_supervised_samples(
    *, split: str, retrieval_samples, graph_samples, metadata_records,
    text_labels, topic_relation_records=None, invalid_diagnostics=None,
):
    """Keep only LLM annotations that are non-empty and match candidates."""
    kept = []
    excluded = list(invalid_diagnostics or [])
    invalid_by_index = {item["original_index"]: item for item in excluded}
    topic_records = topic_relation_records
    if topic_records is None:
        topic_records = [None] * len(retrieval_samples)
    if len(topic_records) != len(retrieval_samples):
        raise ValueError(
            "Topic-relation/retrieval length mismatch before filtering: "
            f"topic_relations={len(topic_records)}, retrieval={len(retrieval_samples)}"
        )
    for index, (retrieval, labels, topic_record) in enumerate(
        zip(retrieval_samples, text_labels, topic_records)
    ):
        candidate_paths = retrieval.get("reasoning_paths", [])
        normalized_candidates = {
            tuple(_normalize_relation_item(value).lower() for value in path)
            for path in candidate_paths
        }
        normalized_label_paths = [
            tuple(_normalize_relation_item(value) for value in path)
            for path in labels
        ]
        valid_labels = [
            path for path in normalized_label_paths
            if tuple(value.lower() for value in path) in normalized_candidates
        ]
        if not valid_labels:
            reason = "empty_llm_annotation" if not labels else "no_matching_candidate_path"
            item = invalid_by_index.get(index, {
                "original_index": index,
                "reason": reason,
                "llm_annotations": normalized_label_paths,
            })
            item["sample_id"] = retrieval.get("id", index)
            if index not in invalid_by_index:
                excluded.append(item)
            continue
        kept.append(index)
    return (
        [graph_samples[i] for i in kept],
        [retrieval_samples[i] for i in kept],
        [
            [
                tuple(_normalize_relation_item(value) for value in path)
                for path in text_labels[i]
            ]
            for i in kept
        ],
        [metadata_records[i] for i in kept],
        None if topic_relation_records is None else [topic_records[i] for i in kept],
        kept,
        excluded,
    )


def apply_sample_limit(
    graph_samples, retrieval_samples, text_labels, metadata_records,
    topic_relation_records, max_samples,
):
    """Apply one prefix limit to every aligned input collection."""
    if max_samples is None:
        return (
            graph_samples, retrieval_samples, text_labels, metadata_records,
            topic_relation_records,
        )
    if max_samples <= 0:
        raise ValueError(f"sample_limit must be strictly positive, got {max_samples!r}.")
    return (
        graph_samples[:max_samples],
        retrieval_samples[:max_samples],
        text_labels[:max_samples],
        metadata_records[:max_samples],
        None if topic_relation_records is None else topic_relation_records[:max_samples],
    )


def _sample_id(record: object, fallback_index: int) -> object:
    if isinstance(record, dict):
        return record.get("id", fallback_index)
    return fallback_index


def validate_inputs(
    *,
    split: str,
    label_path: str | os.PathLike[str],
    annotated_text_labels: AnnotatedTextLabels | None,
    retrieval_samples: Sequence[object],
    metadata_records: Sequence[object],
    graph_samples: Sequence[object],
    allow_empty_labels: bool = False,
) -> None:
    """Validate aligned per-sample inputs before final dataset construction."""
    canonical_split = normalize_split(split)
    context = f"split={canonical_split!r}, label_path={Path(label_path)}"
    if annotated_text_labels is None:
        raise ValueError(f"annotated_text_labels is None ({context}).")
    if not annotated_text_labels and retrieval_samples:
        raise ValueError(f"annotated_text_labels is empty ({context}).")
    if len(annotated_text_labels) != len(retrieval_samples):
        raise ValueError(
            f"Label/retrieval length mismatch ({context}): "
            f"labels={len(annotated_text_labels)}, retrieval={len(retrieval_samples)}."
        )
    if len(metadata_records) != len(retrieval_samples):
        raise ValueError(
            f"Metadata/retrieval length mismatch ({context}): "
            f"metadata={len(metadata_records)}, retrieval={len(retrieval_samples)}."
        )
    if len(graph_samples) != len(retrieval_samples):
        raise ValueError(
            f"Graph/retrieval length mismatch ({context}): "
            f"graphs={len(graph_samples)}, retrieval={len(retrieval_samples)}."
        )
    for sample_index, paths in enumerate(annotated_text_labels):
        if not paths and not allow_empty_labels:
            sample_id = _sample_id(retrieval_samples[sample_index], sample_index)
            raise ValueError(
                f"Empty per-sample annotation ({context}, sample_index={sample_index}, "
                f"sample_id={sample_id!r})."
            )


def _input_signature(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _print_diagnostics(
    *,
    split: str,
    label_path: Path,
    annotated_text_labels: AnnotatedTextLabels,
    retrieval_samples: Sequence[object],
    metadata_records: Sequence[object],
    graph_samples: Sequence[object],
) -> None:
    print("===== POST-PROCESS INPUT SUMMARY =====")
    print("split:", split)
    print("resolved label file path:", label_path)
    print("label path exists:", label_path.exists())
    print("type of text_labels:", type(annotated_text_labels).__name__)
    print("number of labels:", len(annotated_text_labels))
    print("number of retrieval samples:", len(retrieval_samples))
    print("number of metadata samples:", len(metadata_records))
    print("number of graph samples:", len(graph_samples))
    print("number of empty per-sample labels:", sum(not item for item in annotated_text_labels))
    print(
        "first two sample IDs:",
        [_sample_id(item, index) for index, item in enumerate(retrieval_samples[:2])],
    )
    print("first two label values:", annotated_text_labels[:2])
    print("======================================")


def _construct_processed_dataset(
    dataset_class: Callable[..., object],
    *,
    processed_dir: Path,
    split: str,
    graph_samples: Sequence[object],
    topic_relation_path: Path,
    retrieval_samples: Sequence[object],
    metadata_records: Sequence[object],
    annotated_text_labels: AnnotatedTextLabels,
    cache_signature: str,
    force_reprocess: bool,
    topic_relation_records: Sequence[object] | None = None,
    sample_limit: int | None = None,
    requested_sample_limit: int | None = None,
    source_count: int | None = None,
    excluded_count: int = 0,
    exclusion_report: dict | None = None,
    artifact_metadata: dict | None = None,
) -> object:
    """Call the dataset constructor through a testable, explicit boundary."""
    return dataset_class(
        str(processed_dir),
        split,
        graph_samples,
        str(topic_relation_path),
        retrieval_list=retrieval_samples,
        metadata_list=metadata_records,
        **({"labeled_topic_relation_records": topic_relation_records}
           if topic_relation_records is not None else {}),
        text_labels=annotated_text_labels,
        cache_signature=cache_signature,
        force_reprocess=force_reprocess,
        sample_limit=sample_limit,
        requested_sample_limit=requested_sample_limit,
        source_count=source_count,
        excluded_count=excluded_count,
        exclusion_report=exclusion_report,
        artifact_metadata=artifact_metadata,
    )


def _sha256_file(path):
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _load_and_validate_graph_manifest(
    graph_processed_dir,
    split,
    dataset_name,
    embedding_model,
):
    """Load and validate provenance metadata for experimental PyG graphs."""
    manifest_path = (
        Path(graph_processed_dir)
        / split
        / "graph_manifest.json"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing graph manifest for split={split!r}: "
            f"{manifest_path}"
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read graph manifest {manifest_path}: {error}"
        ) from error

    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported graph manifest schema for split={split!r}: "
            f"{manifest.get('schema_version')!r}."
        )

    if manifest.get("split") != split:
        raise ValueError(
            f"Graph manifest split mismatch: "
            f"expected={split!r}, "
            f"found={manifest.get('split')!r}."
        )

    if manifest.get("dataset_name") != dataset_name:
        raise ValueError(
            f"Graph manifest dataset mismatch: "
            f"expected={dataset_name!r}, "
            f"found={manifest.get('dataset_name')!r}."
        )

    if manifest.get("embedding_model") != embedding_model:
        raise ValueError(
            f"Graph manifest embedding mismatch: "
            f"requested={embedding_model!r}, "
            f"found={manifest.get('embedding_model')!r}."
        )

    embedding_dim = manifest.get("embedding_dim")

    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError(
            f"Invalid embedding_dim in graph manifest: "
            f"{embedding_dim!r}."
        )

    sample_count = manifest.get("sample_count")

    if not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError(
            f"Invalid sample_count in graph manifest: "
            f"{sample_count!r}."
        )

    retrieval_path = manifest.get("retrieval_path")
    embedding_path = manifest.get("embedding_path")

    if not retrieval_path or not Path(retrieval_path).is_file():
        raise FileNotFoundError(
            f"Graph manifest retrieval source is missing for split={split!r}: "
            f"{retrieval_path!r}."
        )

    if not embedding_path or not Path(embedding_path).is_file():
        raise FileNotFoundError(
            f"Graph manifest embedding source is missing for split={split!r}: "
            f"{embedding_path!r}."
        )

    expected_retrieval_sha256 = manifest.get("retrieval_sha256")
    expected_embedding_sha256 = manifest.get("embedding_sha256")

    actual_retrieval_sha256 = _sha256_file(retrieval_path)
    actual_embedding_sha256 = _sha256_file(embedding_path)

    if actual_retrieval_sha256 != expected_retrieval_sha256:
        raise ValueError(
            f"Retrieval artifact changed since graph generation "
            f"for split={split!r}."
        )

    if actual_embedding_sha256 != expected_embedding_sha256:
        raise ValueError(
            f"Embedding artifact changed since graph generation "
            f"for split={split!r}."
        )

    return manifest


def _infer_embedding_dim(graph_samples, split):
    """Infer embedding dimension from [head | relation | tail] node features."""
    if not graph_samples:
        raise ValueError(
            f"Cannot infer embedding dimension from empty split {split!r}."
        )

    feature_dim = int(graph_samples[0].x.shape[1])

    if feature_dim % 3 != 0:
        raise ValueError(
            f"Unexpected node feature dimension for split={split!r}: "
            f"{feature_dim} is not divisible by 3."
        )

    return feature_dim // 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process a graph dataset.")

    parser.add_argument(
        "--name",
        default="toy",
        help="Dataset name",
    )

    parser.add_argument(
        "--llm_name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        choices=[
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        help="Annotation model name",
    )

    parser.add_argument(
        "--force-reprocess", "--force_reprocess",
        dest="force_reprocess",
        action="store_true",
        help=(
            "Explicitly replace an incomplete, stale, or "
            "unverifiable cache in the selected output directory."
        ),
    )

    parser.add_argument(
        "--graph-processed-dir",
        type=Path,
        default=None,
        help="Directory containing embedding-dependent PyG graph artifacts.",
    )

    parser.add_argument(
        "--embedding-model",
        dest="embedding_model",
        type=str,
        default=None,
        help=(
            "Embedding model used to generate the graph artifacts. "
            "Required when --graph-processed-dir is provided."
        ),
    )


    parser.add_argument(
        "--embedding-dir",
        default=None,
        help=(
            "Explicit directory containing embedding .pth files. "
            "If omitted, KGDataset uses the legacy "
            "emb/<embedding_model>/ location."
        ),
    )

    parser.add_argument(
        "--output-processed-dir", "--final_dir",
        dest="output_processed_dir",
        type=Path,
        default=None,
        help=(
            "Directory in which the regenerated graph splits are written. "
            "Input pickle files are still read from data_files/<name>/processed."
        ),
    )

    parser.add_argument(
        "--max-samples", "--sample_limit",
        dest="max_samples",
        type=int,
        default=None,
        help="Process only the first N samples of each selected split.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=None,
        help="Dataset splits to process.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)

    args = parser.parse_args()

    if args.split is not None and args.splits is not None:
        parser.error("--split and --splits are mutually exclusive.")
    selected_splits = [args.split] if args.split else (args.splits or ["train", "val", "test"])

    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be a strictly positive integer.")

    from src.dataset.PostProcessedDataset import ProcessedDiskDataset
    from src.dataset.PyG_dataset_disk import KGDataset

    base_dir = Path("data_files") / args.name

    # Retrieval artifacts remain in the shared legacy directory.
    input_processed_dir = base_dir / "processed"

    # PyG graph artifacts may depend on the embedding model.
    graph_processed_dir = (
        args.graph_processed_dir
        if args.graph_processed_dir is not None
        else input_processed_dir
    )

    # Outputs may be redirected to an isolated directory.
    output_processed_dir = (
        args.output_processed_dir
        if args.output_processed_dir is not None
        else input_processed_dir
    )

    if args.graph_processed_dir is not None and args.embedding_model is None:
        parser.error(
            "--graph-processed-dir requires an explicit --embedding-model."
        )

    if args.graph_processed_dir is not None and args.output_processed_dir is None:
        parser.error(
            "--graph-processed-dir requires an explicit --output-processed-dir."
        )

    if output_processed_dir.resolve() == graph_processed_dir.resolve():
        parser.error(
            "Output processed directory must differ from graph input directory."
        )

    if (
        args.max_samples is not None
        and output_processed_dir.resolve() == input_processed_dir.resolve()
    ):
        parser.error(
            "--max-samples cannot be used with the original processed directory. "
            "Provide an isolated --output-processed-dir."
        )

    annotation_root = (
        base_dir
        / "annotated_paths_LLM"
        / args.llm_name
    )

    topic_relation_root = (
        base_dir
        / "relationtargeting"
        / args.llm_name
    )

    print("Shared retrieval directory:", input_processed_dir)
    print("Graph processed directory  :", graph_processed_dir)
    print("Output processed directory :", output_processed_dir)
    print("Selected splits           :", selected_splits)
    print("Maximum samples per split :", args.max_samples)

    for split in selected_splits:
        retrieval_path = (
            input_processed_dir
            / f"{split}_retrieval.pkl"
        )

        metadata_path = (
            graph_processed_dir
            / split
            / f"metadata_{split}.pkl"
        )

        label_path = annotation_root / split

        topic_relation_path = (
            topic_relation_root
            / f"{split}_results.pkl"
        )

        with retrieval_path.open("rb") as retrieval_file:
            retrieval_samples = pickle.load(retrieval_file)

        with metadata_path.open("rb") as metadata_file:
            metadata_records = pickle.load(metadata_file)
        with topic_relation_path.open("rb") as topic_file:
            topic_relation_records = pickle.load(topic_file)

        full_graph_samples = KGDataset(
            root=str(base_dir),
            split=split,
            text_encoder_name=(
                args.embedding_model
                or "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dir=args.embedding_dir,
            graph_processed_dir=str(graph_processed_dir),
        )

        artifact_metadata = None

        if args.graph_processed_dir is not None:
            graph_manifest = _load_and_validate_graph_manifest(
                graph_processed_dir=graph_processed_dir,
                split=split,
                dataset_name=args.name,
                embedding_model=args.embedding_model,
            )

            embedding_dim = _infer_embedding_dim(
                full_graph_samples,
                split,
            )

            if graph_manifest["embedding_dim"] != embedding_dim:
                raise ValueError(
                    f"Graph feature/manifest embedding dimension mismatch "
                    f"for split={split!r}: "
                    f"manifest={graph_manifest['embedding_dim']}, "
                    f"graphs={embedding_dim}."
                )

            if graph_manifest["sample_count"] != len(full_graph_samples):
                raise ValueError(
                    f"Graph manifest sample count mismatch for split={split!r}: "
                    f"manifest={graph_manifest['sample_count']}, "
                    f"graphs={len(full_graph_samples)}."
                )

            artifact_metadata = {
                "dataset_name": args.name,
                "embedding_model": args.embedding_model,
                "embedding_dim": embedding_dim,
                "llm_name": args.llm_name,
                "graph_processed_dir": str(graph_processed_dir),
            }

            print("Artifact metadata          :", artifact_metadata)

        source_retrieval_count = len(retrieval_samples)

        if args.max_samples is None:
            graph_samples = full_graph_samples
        else:
            sample_count = args.max_samples

            if sample_count > source_retrieval_count:
                raise ValueError(
                    f"--max-samples={sample_count} exceeds "
                    f"retrieval size={source_retrieval_count} "
                    f"for split={split!r}."
                )

            if sample_count > len(full_graph_samples):
                raise ValueError(
                    f"--max-samples={sample_count} exceeds "
                    f"available graph count={len(full_graph_samples)} "
                    f"for split={split!r}."
                )

            if sample_count > len(metadata_records):
                raise ValueError(
                    f"--max-samples={sample_count} exceeds "
                    f"metadata count={len(metadata_records)} "
                    f"for split={split!r}."
                )

            if (
                topic_relation_records is not None
                and sample_count > len(topic_relation_records)
            ):
                raise ValueError(
                    f"--max-samples={sample_count} exceeds "
                    f"topic-relation count={len(topic_relation_records)} "
                    f"for split={split!r}."
                )

            retrieval_samples = retrieval_samples[:sample_count]
            metadata_records = metadata_records[:sample_count]

            if topic_relation_records is not None:
                topic_relation_records = topic_relation_records[:sample_count]

            graph_samples = [
                full_graph_samples[index]
                for index in range(sample_count)
            ]

        source_counts = {
            "retrieval": len(retrieval_samples),
            "metadata": len(metadata_records),
            "graphs": len(graph_samples),
            "topic_relations": (
                len(topic_relation_records)
                if topic_relation_records is not None
                else len(retrieval_samples)
            ),
        }

        if len(set(source_counts.values())) != 1:
            raise ValueError(
                f"Source length mismatch after sample limiting "
                f"for split={split!r}: {source_counts}."
            )

        total_samples = len(retrieval_samples)

        print(
            f"Aligned samples for split={split!r}: "
            f"{total_samples} "
            f"(retrieval source={source_retrieval_count})"
        )

        if topic_relation_records is not None and len(topic_relation_records) != len(retrieval_samples):
            raise ValueError(
                f"Topic-relation/retrieval length mismatch after sample_limit "
                f"(split={split!r}): topic_relations={len(topic_relation_records)}, "
                f"retrieval={len(retrieval_samples)}"
            )

        signature_paths = [
            retrieval_path,
            metadata_path,
            topic_relation_path,
        ] + [
            label_path / f"sample_{index}.txt"
            for index in range(len(retrieval_samples))
        ]

        cache_signature = _input_signature(signature_paths)

        manifest_path = (
            output_processed_dir
            / split
            / "postprocess_manifest.json"
        )

        annotation_diagnostics = []
        annotated_text_labels = _collect_annotated_paths(
            label_path,
            split=split,
            expected_samples=len(retrieval_samples),
            allow_extra_samples=args.max_samples is not None,
            allow_invalid=True,
            diagnostics=annotation_diagnostics,
        )

        validate_inputs(
            split=split,
            label_path=label_path,
            annotated_text_labels=annotated_text_labels,
            retrieval_samples=retrieval_samples,
            metadata_records=metadata_records,
            graph_samples=graph_samples,
            allow_empty_labels=True,
        )

        (
            graph_samples,
            retrieval_samples,
            annotated_text_labels,
            metadata_records,
            topic_relation_records,
            valid_indices,
            excluded_samples,
        ) = filter_supervised_samples(
            split=split,
            retrieval_samples=retrieval_samples,
            graph_samples=graph_samples,
            metadata_records=metadata_records,
            text_labels=annotated_text_labels,
            topic_relation_records=topic_relation_records,
            invalid_diagnostics=annotation_diagnostics,
        )
        exclusion_report = {
            "split": split,
            "source_count": total_samples if args.max_samples is None else args.max_samples,
            "kept_count": len(retrieval_samples),
            "excluded_count": len(excluded_samples),
            "excluded_samples": excluded_samples,
        }
        report_path = output_processed_dir / split / "postprocess_excluded_samples.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(exclusion_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if not retrieval_samples:
            print(f"No valid supervised samples for split={split!r}; writing an empty processed split.")

        effective_sample_limit = len(graph_samples) or None

        _print_diagnostics(
            split=split,
            label_path=label_path,
            annotated_text_labels=annotated_text_labels,
            retrieval_samples=retrieval_samples,
            metadata_records=metadata_records,
            graph_samples=graph_samples,
        )

        _construct_processed_dataset(
            ProcessedDiskDataset,
            processed_dir=output_processed_dir,
            split=split,
            graph_samples=graph_samples,
            topic_relation_path=topic_relation_path,
            retrieval_samples=retrieval_samples,
            metadata_records=metadata_records,
            annotated_text_labels=annotated_text_labels,
            topic_relation_records=topic_relation_records,
            cache_signature=cache_signature,
            force_reprocess=args.force_reprocess,
            sample_limit=effective_sample_limit,
            requested_sample_limit=args.max_samples,
            source_count=exclusion_report["source_count"],
            excluded_count=exclusion_report["excluded_count"],
            exclusion_report=exclusion_report,
            artifact_metadata=artifact_metadata,
        )

        print(
            f"Post-processing achieved "
            f"for split={split!r}."
        )

if __name__ == "__main__":
    main()
