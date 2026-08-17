from __future__ import annotations

import json
import hashlib
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from load_post_processed_dataset_v2 import (
    AnnotationParseError,
    _collect_annotated_paths,
    _construct_processed_dataset,
    _load_and_validate_graph_manifest,
    _parse_rational_paths,
    normalize_split,
    validate_inputs,
    filter_supervised_samples,
    apply_sample_limit,
    _normalize_relation_item,
)
from src.dataset.PostProcessedDataset import ProcessedDiskDataset
from src.model.Trainerv3 import Trainer


class AnnotationLoadingTests(unittest.TestCase):
    def test_old_and_arrow_formats_produce_relation_tuples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            (folder / "sample_0.txt").write_text(
                "The rational paths are:\n1. [has_symptom]\n", encoding="utf-8"
            )
            (folder / "sample_1.txt").write_text(
                "The rational path is:\n1. sepsis -> has_symptom -> fever\n",
                encoding="utf-8",
            )

            labels = _collect_annotated_paths(
                folder, split="train", expected_samples=2
            )

        self.assertEqual(labels, [[("has_symptom",)], [("has_symptom",)]])

    def test_llm_relation_targets_are_removed_before_candidate_matching(self) -> None:
        self.assertEqual(
            _normalize_relation_item("memberOfPathway -> Event_R_HSA_8878171"),
            "memberOfPathway",
        )
        result = filter_supervised_samples(
            split="train",
            graph_samples=["graph"],
            retrieval_samples=[{
                "id": "ppi_pw_12070",
                "reasoning_paths": [["hasFunctionalInteractionWith", "memberOfPathway"]],
            }],
            metadata_records=["metadata"],
            text_labels=[[("hasFunctionalInteractionWith", "memberOfPathway -> Event_R_HSA_8878171")]],
            topic_relation_records=[{"response_text": "[relations]"}],
        )
        self.assertEqual(result[5], [0])
        self.assertEqual(result[2], [[("hasFunctionalInteractionWith", "memberOfPathway")]])

    def test_arrow_path_rejects_incomplete_alternation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            (folder / "sample_3.txt").write_text(
                "1. sepsis -> has_symptom\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                AnnotationParseError,
                "split='test'.*sample_3.txt.*sample_index=3.*line=1",
            ):
                _parse_rational_paths(
                    folder, "sample_3.txt", split="test", sample_index=3
                )

    def test_missing_directory_is_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing = Path(temporary_dir) / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "split='train'.*missing"):
                _collect_annotated_paths(
                    missing, split="train", expected_samples=1
                )

    def test_missing_sample_index_does_not_shift_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            for index in (0, 2):
                (folder / f"sample_{index}.txt").write_text(
                    f"1. [relation_{index}]\n", encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, r"missing=\[1\]"):
                _collect_annotated_paths(
                    folder, split="train", expected_samples=3
                )

    def test_valid_file_without_path_is_an_empty_label_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            (folder / "sample_0.txt").write_text("No path here.\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "Empty annotation.*sample_0.txt.*sample_index=0"
            ):
                _collect_annotated_paths(
                    folder, split="val", expected_samples=1
                )

    def test_explicit_indices_preserve_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            (folder / "sample_1.txt").write_text("1. [second]\n", encoding="utf-8")
            (folder / "sample_0.txt").write_text("1. [first]\n", encoding="utf-8")
            labels = _collect_annotated_paths(
                folder, split="train", expected_samples=2
            )
        self.assertEqual(labels, [[("first",)], [("second",)]])

    def test_split_aliases(self) -> None:
        self.assertEqual(normalize_split("val"), "val")
        self.assertEqual(normalize_split("valid"), "val")
        self.assertEqual(normalize_split("validation"), "val")

    def test_loaded_labels_are_passed_unchanged_to_constructor(self) -> None:
        captured: dict[str, object] = {}

        def fake_dataset(*args, **kwargs):
            captured.update(kwargs)
            return object()

        labels = [[("has_symptom",)]]
        _construct_processed_dataset(
            fake_dataset,
            processed_dir=Path("processed"),
            split="train",
            graph_samples=[object()],
            topic_relation_path=Path("relations.pkl"),
            retrieval_samples=[{"id": 42}],
            metadata_records=[[{"h_id": 0, "relation_id": 0, "t_id": 1}]],
            annotated_text_labels=labels,
            cache_signature="signature",
            force_reprocess=False,
        )
        self.assertIs(captured["text_labels"], labels)

    def test_constructor_receives_effective_count_after_exclusion(self) -> None:
        captured: dict[str, object] = {}

        def fake_dataset(*args, **kwargs):
            captured.update(kwargs)
            return object()

        kept_graphs = [object() for _ in range(9)]
        _construct_processed_dataset(
            fake_dataset,
            processed_dir=Path("processed"),
            split="train",
            graph_samples=kept_graphs,
            topic_relation_path=Path("relations.pkl"),
            retrieval_samples=[{"id": i} for i in range(9)],
            metadata_records=[[{}] for _ in range(9)],
            annotated_text_labels=[[('r',)] for _ in range(9)],
            topic_relation_records=[{"response_text": "[r]"} for _ in range(9)],
            cache_signature="signature",
            force_reprocess=True,
            sample_limit=len(kept_graphs),
            source_count=10,
            excluded_count=1,
        )
        self.assertEqual(captured["sample_limit"], 9)
        self.assertEqual(captured["source_count"], 10)
        self.assertEqual(captured["excluded_count"], 1)


class InputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.common = {
            "split": "train",
            "label_path": Path("labels/train"),
            "retrieval_samples": [{"id": 7}],
            "metadata_records": [[{}]],
            "graph_samples": [object()],
        }

    def test_none_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "is None.*split='train'"):
            validate_inputs(annotated_text_labels=None, **self.common)

    def test_empty_outer_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "is empty.*split='train'"):
            validate_inputs(annotated_text_labels=[], **self.common)

    def test_empty_per_sample_label_reports_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_index=0, sample_id=7"):
            validate_inputs(annotated_text_labels=[[]], **self.common)

    def test_label_retrieval_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "Label/retrieval length mismatch"):
            validate_inputs(
                annotated_text_labels=[[('one',)], [('two',)]], **self.common
            )

    def test_metadata_retrieval_length_mismatch(self) -> None:
        values = dict(self.common)
        values["metadata_records"] = []
        with self.assertRaisesRegex(ValueError, "Metadata/retrieval length mismatch"):
            validate_inputs(annotated_text_labels=[[('one',)]], **values)

    def test_graph_retrieval_length_mismatch(self) -> None:
        values = dict(self.common)
        values["graph_samples"] = []
        with self.assertRaisesRegex(ValueError, "Graph/retrieval length mismatch"):
            validate_inputs(annotated_text_labels=[[('one',)]], **values)

    def test_empty_llm_annotation_is_excluded_without_fallback_to_candidates(self) -> None:
        result = filter_supervised_samples(
            split="train",
            graph_samples=["graph-0", "graph-1"],
            retrieval_samples=[
                {"id": "empty", "reasoning_paths": [["r"]]},
                {"id": "kept", "reasoning_paths": [["r"]]},
            ],
            metadata_records=["meta-0", "meta-1"],
            text_labels=[[], [("r",)]],
            topic_relation_records=[{"response_text": "[]"}, {"response_text": "[r]"}],
        )
        graphs, retrieval, labels, metadata, relations, valid_indices, excluded = result
        self.assertEqual(graphs, ["graph-1"])
        self.assertEqual([item["id"] for item in retrieval], ["kept"])
        self.assertEqual(labels, [[("r",)]])
        self.assertEqual(metadata, ["meta-1"])
        self.assertEqual(len(relations), 1)
        self.assertEqual(valid_indices, [1])
        self.assertEqual(excluded[0]["reason"], "empty_llm_annotation")

    def test_sample_limit_applies_to_topic_relations_with_same_prefix(self) -> None:
        values = apply_sample_limit(
            list(range(12307)), list(range(12307)), list(range(12307)),
            list(range(12307)), list(range(12307)), 10,
        )
        self.assertEqual([len(value) for value in values], [10, 10, 10, 10, 10])


class CacheAndProcessingTests(unittest.TestCase):
    @staticmethod
    def _fixture_values():
        graph = Data(
            x=torch.zeros((1, 3)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            topic_candidates=torch.ones((1, 1), dtype=torch.long),
            one_hop_neighbors=[[]],
        )
        retrieval = [{
            "id": 9,
            "reasoning_paths": [["has_symptom"]],
            "translated_paths": ["sepsis -> has_symptom -> fever"],
            "id2entities": {0: "sepsis", 1: "fever"},
            "id2relations": {0: "has_symptom"},
        }]
        metadata = [[{"h_id": 0, "relation_id": 0, "t_id": 1}]]
        return [graph], retrieval, metadata, [[("has_symptom",)]]

    def test_processing_writes_manifest_and_valid_cache_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            relation_path = root / "relations.pkl"
            relation_path.write_bytes(
                pickle.dumps([{"response_text": "<Solution> [has_symptom]"}])
            )
            graphs, retrieval, metadata, labels = self._fixture_values()
            first = ProcessedDiskDataset(
                str(root), "train", graphs, str(relation_path), retrieval,
                labels, metadata, cache_signature="abc"
            )
            graph_mtime = (root / "train" / "data_0.pt").stat().st_mtime_ns
            second = ProcessedDiskDataset(
                str(root), "train", graphs, str(relation_path), retrieval,
                labels, metadata, cache_signature="abc"
            )
            load_only = ProcessedDiskDataset(str(root), "train")

            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertEqual(len(load_only), 1)
            self.assertEqual(
                (root / "train" / "data_0.pt").stat().st_mtime_ns, graph_mtime
            )
            manifest = json.loads(
                (root / "train" / "postprocess_manifest.json").read_text()
            )
            self.assertEqual(manifest["input_signature"], "abc")

    def test_unverifiable_existing_cache_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            split_dir = root / "train"
            split_dir.mkdir()
            torch.save(Data(x=torch.zeros((1, 1))), split_dir / "data_0.pt")
            graphs, retrieval, metadata, labels = self._fixture_values()
            with self.assertRaisesRegex(RuntimeError, "unverifiable cache.*force"):
                ProcessedDiskDataset(
                    str(root), "train", graphs, str(root / "missing.pkl"),
                    retrieval, labels, metadata, cache_signature="abc"
                )

    def test_stale_cache_requires_force_and_force_regenerates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            split_dir = root / "train"
            split_dir.mkdir()
            stale_graph = Data(
                x=torch.zeros((1, 1)),
                path_label=torch.zeros((1, 1), dtype=torch.long),
                topic_labels=torch.ones((1, 1), dtype=torch.long),
                topic_candidates=torch.ones((1, 1), dtype=torch.long),
            )
            torch.save(stale_graph, split_dir / "data_0.pt")
            (split_dir / "postprocess_manifest.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "split": "train",
                    "sample_count": 1,
                    "input_signature": "old",
                }),
                encoding="utf-8",
            )
            relation_path = root / "relations.pkl"
            relation_path.write_bytes(
                pickle.dumps([{"response_text": "<Solution> [has_symptom]"}])
            )
            graphs, retrieval, metadata, labels = self._fixture_values()
            preserved_source = split_dir / "metadata_train.pkl"
            preserved_source.write_bytes(b"preserve this source metadata")
            with self.assertRaisesRegex(RuntimeError, "stale cache.*force"):
                ProcessedDiskDataset(
                    str(root), "train", graphs, str(relation_path), retrieval,
                    labels, metadata, cache_signature="new"
                )

            ProcessedDiskDataset(
                str(root), "train", graphs, str(relation_path), retrieval,
                labels, metadata, cache_signature="new", force_reprocess=True
            )
            manifest = json.loads(
                (split_dir / "postprocess_manifest.json").read_text()
            )
            self.assertEqual(manifest["input_signature"], "new")
            self.assertEqual(
                preserved_source.read_bytes(), b"preserve this source metadata"
            )

    def test_topic_relation_question_alignment_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            relation_path = root / "relations.pkl"
            relation_path.write_bytes(pickle.dumps([{
                "question": "a different question",
                "response_text": "<Solution> [has_symptom]",
            }]))
            graphs, retrieval, metadata, labels = self._fixture_values()
            retrieval[0]["question"] = "expected question"
            with self.assertRaisesRegex(ValueError, "Question alignment mismatch"):
                ProcessedDiskDataset(
                    str(root), "train", graphs, str(relation_path), retrieval,
                    labels, metadata, cache_signature="abc"
                )

    def test_incomplete_existing_cache_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            split_dir = root / "train"
            split_dir.mkdir()
            torch.save(Data(x=torch.zeros((1, 1))), split_dir / "data_1.pt")
            graphs, retrieval, metadata, labels = self._fixture_values()
            with self.assertRaisesRegex(RuntimeError, "incomplete cache.*force"):
                ProcessedDiskDataset(
                    str(root), "train", graphs, str(root / "missing.pkl"),
                    retrieval, labels, metadata, cache_signature="abc"
                )

    def test_processing_failure_does_not_save_unmodified_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            relation_path = root / "relations.pkl"
            relation_path.write_bytes(
                pickle.dumps([{"response_text": "<Solution> [has_symptom]"}])
            )
            graphs, retrieval, metadata, labels = self._fixture_values()
            retrieval[0]["reasoning_paths"] = [["different_relation"]]
            with self.assertRaisesRegex(RuntimeError, "sample_index=0"):
                ProcessedDiskDataset(
                    str(root), "train", graphs, str(relation_path), retrieval,
                    labels, metadata, cache_signature="abc"
                )
            self.assertFalse((root / "train" / "data_0.pt").exists())
            self.assertFalse(
                (root / "train" / "postprocess_manifest.json").exists()
            )

    def test_mid_split_failure_preserves_complete_legacy_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            split_dir = root / "train"
            split_dir.mkdir()
            legacy_manifest = json.dumps({
                "schema_version": 1,
                "split": "train",
                "sample_count": 2,
                "input_signature": "legacy",
            }) + "\n"
            (split_dir / "postprocess_manifest.json").write_text(
                legacy_manifest, encoding="utf-8"
            )
            for index in range(2):
                torch.save(
                    Data(
                        x=torch.full((1, 1), float(index + 10)),
                        path_label=torch.zeros((1, 1), dtype=torch.long),
                        topic_labels=torch.ones((1, 1), dtype=torch.long),
                        topic_candidates=torch.ones((1, 1), dtype=torch.long),
                    ),
                    split_dir / f"data_{index}.pt",
                )
            original_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in split_dir.iterdir()
            }

            relation_path = root / "relations.pkl"
            relation_path.write_bytes(pickle.dumps([
                {"response_text": "<Solution> [has_symptom]"},
                {"response_text": "<Solution> [has_symptom]"},
            ]))
            graphs, retrieval, metadata, labels = self._fixture_values()
            graphs = [graphs[0], graphs[0].clone()]
            retrieval = [retrieval[0], dict(retrieval[0], id=10)]
            retrieval[1]["reasoning_paths"] = [["different_relation"]]
            metadata = [metadata[0], metadata[0]]
            labels = [labels[0], labels[0]]

            with self.assertRaisesRegex(RuntimeError, "sample_index=1"):
                ProcessedDiskDataset(
                    str(root), "train", graphs, str(relation_path), retrieval,
                    labels, metadata, cache_signature="new", force_reprocess=True
                )

            final_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in split_dir.iterdir()
            }
            self.assertEqual(final_hashes, original_hashes)
            self.assertEqual(
                (split_dir / "postprocess_manifest.json").read_text(),
                legacy_manifest,
            )
            self.assertEqual(list(root.glob(".train.staging-*")), [])


class TrainerLossSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = Trainer(
            model_type="MLP",
            num_layers=1,
            in_dims=3,
            emb_size=1,
            hidden_dims=2,
            out_dims=2,
            batch_norm=False,
            dropout=0.0,
            max_depth=3,
            device="cpu",
        )

    def test_ignore_zero_topic_and_total_loss_semantics(self) -> None:
        with_ignored = self.trainer._path_steps_from_column(
            torch.tensor([0, -1, -1], dtype=torch.long)
        )
        without_ignored = self.trainer._path_steps_from_column(
            torch.tensor([0], dtype=torch.long)
        )
        self.assertEqual(with_ignored, without_ignored)

        path_logits = torch.tensor([0.0, 0.0], requires_grad=True)
        path_loss = self.trainer._path_cross_entropy(
            path_logits, target_index=0, context="unit test"
        )
        changed_path_loss = self.trainer._path_cross_entropy(
            torch.tensor([2.0, 0.0]), target_index=0, context="unit test"
        )
        self.assertGreater(path_loss.item(), 0.0)
        self.assertLess(changed_path_loss.item(), path_loss.item())

        topic_loss, positive_targets, negative_targets = (
            self.trainer._topic_binary_loss(
                torch.tensor(0.0, requires_grad=True),
                torch.tensor([0.0, 0.0], requires_grad=True),
            )
        )
        changed_topic_loss, _, _ = self.trainer._topic_binary_loss(
            torch.tensor(2.0), torch.tensor([-2.0, -2.0])
        )
        self.assertEqual(positive_targets.tolist(), [1.0])
        self.assertEqual(negative_targets.tolist(), [0.0, 0.0])
        self.assertLess(changed_topic_loss.item(), topic_loss.item())

        total_loss = path_loss + topic_loss
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()
        self.assertTrue(torch.any(path_logits.grad != 0))

    def test_out_of_range_path_target_has_clear_error(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"target=2, valid_range=\[0, 1\]"
        ):
            self.trainer._path_cross_entropy(
                torch.tensor([0.0, 0.0]),
                target_index=2,
                context="unit test",
            )

    def test_path_step_ordinals_must_be_contiguous_and_in_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous from 0"):
            self.trainer._path_steps_from_column(torch.tensor([0, 2, -1]))
        with self.assertRaisesRegex(ValueError, "exceeding.*max_depth=3"):
            self.trainer._path_steps_from_column(torch.tensor([0, 1, 2, 3]))

    def _decision_statistics(self, decision_type: str, loss_value: float):
        statistics = self.trainer._new_path_statistics()
        statistics["supervised_path_count"] = 1
        logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
        loss = torch.tensor(loss_value, dtype=torch.float32)
        self.trainer._record_path_decision(
            statistics, decision_type, logits, loss
        )
        return statistics

    def _batch_statistics(self, *path_statistics):
        statistics = self.trainer._merge_path_statistics(path_statistics)
        self.trainer._finish_path_batch(statistics)
        return statistics

    def test_unsupervised_batch_does_not_dilute_epoch_path_loss(self) -> None:
        supervised = self._batch_statistics(
            self._decision_statistics("stop", 2.5)
        )
        unsupervised = self._batch_statistics()
        unsupervised_summary = self.trainer._summarize_path_statistics(
            unsupervised
        )

        summary = self.trainer._summarize_path_statistics(
            self.trainer._merge_path_statistics([supervised, unsupervised])
        )

        self.assertEqual(summary["mean_path_loss"], 2.5)
        self.assertEqual(summary["batches_with_path_supervision"], 1)
        self.assertEqual(summary["batches_without_path_supervision"], 1)
        self.assertIsNone(unsupervised_summary["mean_path_loss"])
        self.assertIsNone(unsupervised_summary["mean_transition_loss"])
        self.assertIsNone(unsupervised_summary["mean_stop_loss"])

    def test_reported_path_loss_is_independent_of_batch_grouping(self) -> None:
        paths = [
            self._decision_statistics("stop", 1.0),
            self._decision_statistics("stop", 3.0),
            self._decision_statistics("transition", 5.0),
        ]
        first_grouping = [
            self._batch_statistics(paths[0], paths[1]),
            self._batch_statistics(paths[2]),
        ]
        second_grouping = [
            self._batch_statistics(paths[0]),
            self._batch_statistics(paths[1], paths[2]),
        ]

        first = self.trainer._summarize_path_statistics(
            self.trainer._merge_path_statistics(first_grouping)
        )
        second = self.trainer._summarize_path_statistics(
            self.trainer._merge_path_statistics(second_grouping)
        )

        self.assertEqual(first["mean_path_loss"], 3.0)
        self.assertEqual(first["mean_path_loss"], second["mean_path_loss"])
        self.assertEqual(first["transition_decision_count"], 1)
        self.assertEqual(first["stop_decision_count"], 2)

    def test_one_edge_path_records_only_stop_decision(self) -> None:
        _, statistics = self.trainer._path_loss_for_column(
            column_vals=torch.tensor([0, -1]),
            node_emb=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            question_emb=torch.zeros(2),
            one_hop_neighbors=[[1], [0]],
            stop_emb=torch.tensor([[0.5, 0.5]]),
            context_label="one-edge test",
        )
        summary = self.trainer._summarize_path_statistics(statistics)

        self.assertEqual(summary["supervised_path_count"], 1)
        self.assertEqual(summary["transition_decision_count"], 0)
        self.assertEqual(summary["stop_decision_count"], 1)
        self.assertIsNone(summary["mean_transition_loss"])
        self.assertIsNotNone(summary["mean_stop_loss"])

    def test_multi_edge_path_records_transition_and_stop_decisions(self) -> None:
        path_loss, statistics = self.trainer._path_loss_for_column(
            column_vals=torch.tensor([0, 1]),
            node_emb=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            question_emb=torch.zeros(2),
            one_hop_neighbors=[[1], [0]],
            stop_emb=torch.tensor([[0.5, 0.5]]),
            context_label="multi-edge test",
        )
        summary = self.trainer._summarize_path_statistics(statistics)

        self.assertTrue(torch.isfinite(path_loss))
        self.assertEqual(summary["supervised_path_count"], 1)
        self.assertEqual(summary["transition_decision_count"], 1)
        self.assertEqual(summary["stop_decision_count"], 1)
        self.assertIsNotNone(summary["mean_transition_loss"])
        self.assertIsNotNone(summary["mean_stop_loss"])

    def test_fit_history_uses_only_observed_path_decisions(self) -> None:
        supervised = self._batch_statistics(
            self._decision_statistics("stop", 2.5)
        )
        unsupervised = self._batch_statistics()
        batch_statistics = iter([supervised, unsupervised])

        def fake_train_step(batch, scaler, pathtrainingstart):
            self.trainer.last_train_step_path_statistics = next(batch_statistics)
            return 3.0, 1.0, 2.5 if batch == "supervised" else 0.0

        def fake_evaluate(*args, **kwargs):
            self.trainer.validation_grad_enabled_observations.append(False)
            self.trainer.last_evaluation_path_statistics = (
                self.trainer._summarize_path_statistics(supervised)
            )
            return 1.0, 2.5, 0.5, 0.5

        self.trainer.epochs = 1
        self.trainer.train_step = fake_train_step
        self.trainer.evaluate = fake_evaluate
        history = self.trainer.fit(
            ["supervised", "unsupervised"],
            val_dataloader=["validation"],
            save_dir=None,
            pathtrainingstart=True,
        )
        epoch = history["epochs"][0]

        self.assertEqual(epoch["train_path_loss"], 2.5)
        self.assertEqual(epoch["train_supervised_path_count"], 1)
        self.assertEqual(epoch["train_stop_decision_count"], 1)
        self.assertEqual(epoch["train_transition_decision_count"], 0)
        self.assertEqual(epoch["train_batches_with_path_supervision"], 1)
        self.assertEqual(epoch["train_batches_without_path_supervision"], 1)
        self.assertIsNone(epoch["train_mean_transition_loss"])




class GraphManifestValidationTests(unittest.TestCase):
    DATASET_NAME = "OntoOmicsKG_step2"
    SPLIT = "train"
    EMBEDDING_MODEL = "dmis-lab/biobert-base-cased-v1.2"

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
        return digest.hexdigest()

    def _create_fixture(self, temporary_dir):
        root = Path(temporary_dir)

        graph_processed_dir = root / "graph_processed"
        split_dir = graph_processed_dir / self.SPLIT
        split_dir.mkdir(parents=True)

        retrieval_path = root / "train_retrieval.pkl"
        embedding_path = root / "train.pth"

        retrieval_content = b"fake retrieval artifact"
        embedding_content = b"fake embedding artifact"

        retrieval_path.write_bytes(retrieval_content)
        embedding_path.write_bytes(embedding_content)

        manifest = {
            "schema_version": 1,
            "dataset_name": self.DATASET_NAME,
            "split": self.SPLIT,
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_dim": 768,
            "sample_count": 3,
            "retrieval_path": str(retrieval_path),
            "retrieval_sha256": self._sha256_file(retrieval_path),
            "embedding_path": str(embedding_path),
            "embedding_sha256": self._sha256_file(embedding_path),
        }

        manifest_path = split_dir / "graph_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        return {
            "graph_processed_dir": graph_processed_dir,
            "retrieval_path": retrieval_path,
            "embedding_path": embedding_path,
            "manifest": manifest,
        }

    def _validate(self, graph_processed_dir):
        return _load_and_validate_graph_manifest(
            graph_processed_dir=graph_processed_dir,
            split=self.SPLIT,
            dataset_name=self.DATASET_NAME,
            embedding_model=self.EMBEDDING_MODEL,
        )

    def test_valid_graph_manifest_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            fixture = self._create_fixture(temporary_dir)

            result = self._validate(
                fixture["graph_processed_dir"]
            )

            self.assertEqual(
                result,
                fixture["manifest"],
            )

    def test_mutated_retrieval_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            fixture = self._create_fixture(temporary_dir)

            fixture["retrieval_path"].write_bytes(
                b"mutated retrieval artifact"
            )

            with self.assertRaisesRegex(
                ValueError,
                "Retrieval artifact changed since graph generation",
            ):
                self._validate(
                    fixture["graph_processed_dir"]
                )

    def test_mutated_embedding_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            fixture = self._create_fixture(temporary_dir)

            fixture["embedding_path"].write_bytes(
                b"mutated embedding artifact"
            )

            with self.assertRaisesRegex(
                ValueError,
                "Embedding artifact changed since graph generation",
            ):
                self._validate(
                    fixture["graph_processed_dir"]
                )


class TrainingEntrypointTests(unittest.TestCase):
    def test_smoke_controls_are_explicit_cli_arguments(self) -> None:
        from main import build_arg_parser

        args = build_arg_parser().parse_args([
            "--dataset_name", "toy",
            "--device", "-1",
            "--epochs", "2",
            "--batch_size", "2",
            "--num_workers", "0",
            "--seed", "17",
            "--output_dir", "smoke_runs/test",
        ])
        self.assertEqual(args.dataset_name, "toy")
        self.assertEqual(args.device, -1)
        self.assertEqual(args.num_workers, 0)
        self.assertEqual(args.seed, 17)
        self.assertEqual(args.output_dir, "smoke_runs/test")

    def test_checkpoint_payload_is_reloadable(self) -> None:
        trainer = Trainer(
            model_type="MLP",
            num_layers=1,
            in_dims=3,
            emb_size=1,
            hidden_dims=2,
            out_dims=2,
            batch_norm=False,
            device="cpu",
        )
        payload = trainer._checkpoint_payload(
            epoch=2,
            metrics={"train_total_loss": 1.25},
        )
        self.assertEqual(payload["epoch"], 2)
        self.assertIn("model_state_dict", payload)
        self.assertIn("optimizer_state_dict", payload)
        self.assertEqual(payload["metrics"]["train_total_loss"], 1.25)

    def test_default_gcn_weights_receive_toy_training_gradient(self) -> None:
        import random

        from torch_geometric.loader import DataLoader

        torch.manual_seed(17)
        random.seed(17)

        source_processed_dir = Path("data_files/toy/processed")
        source_train_dir = source_processed_dir / "train"

        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_processed_dir = (
                Path(temporary_dir) / "processed"
            )
            temporary_train_dir = (
                temporary_processed_dir / "train"
            )

            shutil.copytree(
                source_train_dir,
                temporary_train_dir,
            )

            graph_files = sorted(
                temporary_train_dir.glob("data_*.pt"),
                key=lambda path: int(
                    path.stem.split("_")[1]
                ),
            )

            valid_graphs = []

            for graph_path in graph_files:
                graph = torch.load(
                    graph_path,
                    map_location="cpu",
                    weights_only=False,
                )

                if not hasattr(graph, "path_label"):
                    continue

                if not hasattr(graph, "topic_labels"):
                    continue

                if not hasattr(graph, "topic_candidates"):
                    continue

                if not torch.any(graph.path_label == 0):
                    continue

                valid_graphs.append(graph)

                if len(valid_graphs) == 2:
                    break

            self.assertEqual(
                len(valid_graphs),
                2,
                "Expected at least two valid supervised toy graphs.",
            )

            for graph_path in graph_files:
                graph_path.unlink()

            for sample_index, graph in enumerate(valid_graphs):
                torch.save(
                    graph,
                    temporary_train_dir / f"data_{sample_index}.pt",
                )

            sample_count = len(valid_graphs)

            manifest = {
                "schema_version": 1,
                "split": "train",
                "sample_count": sample_count,
                "input_signature": "toy_training_gradient_fixture",
                "complete_dataset": True,
                "sample_limit": None,
                "source_count": sample_count,
                "kept_count": sample_count,
                "excluded_count": 0,
            }

            manifest_path = (
                temporary_train_dir
                / "postprocess_manifest.json"
            )

            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            dataset = ProcessedDiskDataset(
                str(temporary_processed_dir),
                "train",
            )

            batch = next(
                iter(
                    DataLoader(
                        dataset,
                        batch_size=2,
                        shuffle=False,
                    )
                )
            )

            trainer = Trainer(
                model_type="GCN",
                num_layers=3,
                in_dims=1152,
                emb_size=384,
                hidden_dims=512,
                out_dims=512,
                batch_norm=False,
                dropout=0.2,
                device="cpu",
            )

            trainer.train_step(
                batch,
                torch.amp.GradScaler("cpu"),
                pathtrainingstart=True,
            )

            weight_gradients = [
                layer.lin.weight.grad
                for layer in trainer.convs
            ]

            self.assertTrue(
                any(
                    gradient is not None
                    and torch.isfinite(gradient).all()
                    and torch.any(gradient != 0)
                    for gradient in weight_gradients
                ),
                (
                    "Expected at least one GCN weight "
                    "to receive a finite non-zero gradient."
                ),
            )


if __name__ == "__main__":
    unittest.main()
