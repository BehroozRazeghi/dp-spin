from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.assign import topK_sparse_assignments
from src.claims import claims_from_plan_template, render_verified_claims, verify_structured_claims
from src.dp_mechanisms import (
    analytic_gaussian_sigma,
    exponential_mechanism_topL_select,
    selection_epsilon_schedule,
    sequential_em_utility_bound,
    verify_gaussian_calibration,
    sparse_private_topL_release,
)
from src.matched_protocol import validate_matched_configs
from src.human_eval import create_blinded_batch, evaluate_annotations
from src.repeated_trials import paired_bootstrap_difference
from src.run_experiment import run as run_experiment
from src.utils import read_jsonl
from src.utility_decomposition import radial_l1_clipping_distortion
from src.verifier_benchmark import evaluate, generate_cases
from src.verify import verify_summary


class PipelineExtensionTests(unittest.TestCase):
    def test_strict_text_verifier_uses_marker_local_support_bins(self):
        atoms = [
            {"atom_id": 1, "label": "alpha", "description": "alpha description"},
            {"atom_id": 2, "label": "beta", "description": "beta description"},
            {"atom_id": 3, "label": "gamma", "description": "gamma description"},
        ]
        plan = {
            "selected_atoms": [
                {"atom_id": 1, "label": "alpha", "rank": 1, "noisy_mass": 5.0, "support_bin": "high"},
                {"atom_id": 2, "label": "beta", "rank": 2, "noisy_mass": 3.0, "support_bin": "moderate"},
                {"atom_id": 3, "label": "gamma", "rank": 3, "noisy_mass": 1.0, "support_bin": "low"},
            ]
        }
        summary = "[atom:1] alpha (high), [atom:2] beta (moderate), [atom:3] gamma (low)."
        result = verify_summary(summary, plan, atoms, strict_markers=True)
        self.assertTrue(result["accept"])
        self.assertEqual(result["unsupported_support_bins"], [])
        self.assertEqual(len(result["supported_support_bins"]), 3)

    def test_strict_text_verifier_uar_uses_canonical_markers_only(self):
        atoms = [
            {"atom_id": 1, "label": "alpha", "description": "mentions beta as public descriptive text"},
            {"atom_id": 2, "label": "beta", "description": "beta description"},
        ]
        plan = {
            "selected_atoms": [
                {"atom_id": 1, "label": "alpha", "description": "mentions beta as public descriptive text", "rank": 1, "noisy_mass": 5.0, "support_bin": "high"},
            ]
        }
        summary = "[atom:1] alpha (high): mentions beta as public descriptive text."
        result = verify_summary(summary, plan, atoms, strict_markers=True)
        self.assertEqual(result["mentioned_marker_atom_ids"], [1])
        self.assertEqual(result["mentioned_literal_label_atom_ids"], [1, 2])
        self.assertEqual(result["mentioned_atom_ids"], [1])
        self.assertEqual(result["UAR"], 0.0)
        self.assertTrue(result["accept"])

    def test_ucr_reports_comparison_counts_when_no_comparison_is_detected(self):
        atoms = [{"atom_id": 1, "label": "alpha", "description": "alpha description"}]
        plan = {"selected_atoms": [{"atom_id": 1, "label": "alpha", "rank": 1, "noisy_mass": 5.0, "support_bin": "high"}]}
        result = verify_summary("[atom:1] alpha (high).", plan, atoms, strict_markers=True)
        self.assertEqual(result["UCR"], 0.0)
        self.assertEqual(result["comparison_count"], 0)
        self.assertEqual(result["supported_comparison_count"], 0)
        self.assertEqual(result["unsupported_comparison_count"], 0)

    def test_abstention_preserves_bounds(self):
        records = [{"text": "apple"}, {"text": "completely unrelated xyz"}]
        atoms = [{"atom_id": 0, "label": "apple", "description": "apple fruit"}, {"atom_id": 1, "label": "banana", "description": "banana fruit"}]
        C, V, d = topK_sparse_assignments(records, atoms, K=1, similarity_threshold=0.2, abstention_mode="zero", return_diagnostics=True)
        masses = np.asarray(V.sum(axis=1)).ravel()
        self.assertTrue(np.all(masses <= 1 + 1e-12))
        self.assertGreaterEqual(d["n_abstained"], 1)
        np.testing.assert_allclose(C, np.asarray(V.sum(axis=0)).ravel())

    def test_structured_verifier_and_renderer(self):
        plan = {"selected_atoms": [{"atom_id": 0, "label": "A", "rank": 1, "raw_noisy_mass": 5, "noisy_mass": 5, "support_bin": "high"}, {"atom_id": 1, "label": "B", "rank": 2, "raw_noisy_mass": 3, "noisy_mass": 3, "support_bin": "moderate"}]}
        claims = claims_from_plan_template(plan)
        result = verify_structured_claims(claims, plan)
        self.assertTrue(result["accept"])
        text = render_verified_claims(claims, plan)
        self.assertIn("A", text)
        bad = verify_structured_claims([{"claim_type": "theme", "atom_id": 999}], plan)
        self.assertFalse(bad["accept"])

    def test_verifier_benchmark_has_zero_schema_false_accepts(self):
        result = evaluate(generate_cases(240, 0))
        self.assertEqual(result["false_accept_count"], 0)
        self.assertEqual(result["false_reject_count"], 0)

    def test_gaussian_calibration_verifies(self):
        sigma = analytic_gaussian_sigma(2.0, 1e-6, 1.5)
        report = verify_gaussian_calibration(2.0, 1e-6, sigma, 1.5)
        self.assertTrue(report["valid"])

    def test_selection_allocations_sum_to_budget(self):
        for mode in ["uniform", "front_loaded", "theory_optimal"]:
            schedule = selection_epsilon_schedule(5, 2.0, 20, mode, 0.05)
            self.assertAlmostEqual(float(schedule.sum()), 2.0)
            bound = sequential_em_utility_bound(20, 5, 2.0, allocation=mode)
            self.assertEqual(len(bound["alpha_t"]), 5)
            _, meta = exponential_mechanism_topL_select(np.arange(20.0), 5, 2.0, rng=np.random.default_rng(0), allocation=mode)
            self.assertEqual(meta["selection_budget_allocation"], mode)


    def test_sparse_release_metadata_matches_value_mechanism(self):
        sketch = np.asarray([5.0, 3.0, 1.0, 0.5])
        _, metadata = sparse_private_topL_release(
            sketch,
            L=2,
            epsilon=2.0,
            delta=0.0,
            value_mechanism="laplace",
            value_epsilon_fraction=0.5,
            selection_budget_allocation="theory_optimal",
            rng=np.random.default_rng(0),
        )
        self.assertIn("Laplace values", metadata["mechanism_display_name"])
        self.assertEqual(metadata["delta"], 0.0)
        self.assertIsNone(metadata["atom_admission_release"]["epsilon_per_round"])
        self.assertAlmostEqual(
            sum(metadata["atom_admission_release"]["epsilon_schedule"]),
            metadata["epsilon_selection"],
        )

    def test_end_to_end_artifact_boundary_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records_path = root / "records.jsonl"
            atoms_path = root / "atoms.json"
            records = [
                {"id": "r1", "text": "refund delayed"},
                {"id": "r2", "text": "login problem"},
                {"id": "r3", "text": "refund problem"},
            ]
            atoms = [
                {"atom_id": 0, "label": "refund", "description": "refund delayed"},
                {"atom_id": 1, "label": "account access", "description": "login problem"},
            ]
            records_path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            atoms_path.write_text(json.dumps(atoms), encoding="utf-8")
            outdir = run_experiment({
                "experiment_name": "boundary_test",
                "records": str(records_path),
                "atoms": str(atoms_path),
                "output_dir": str(root / "outputs"),
                "protected_unit": "record",
                "n_records": 3,
                "record_selection": "first",
                "public_denominator": 3,
                "m": 2,
                "K": 1,
                "L": 2,
                "gamma": 10.0,
                "mechanism": "gaussian",
                "epsilon": 1.0,
                "delta": 1e-6,
                "adjacency": "adddrop",
                "decoder_backend": "template",
                "decoder_contract_mode": "structured",
                "assignment_show_progress": False,
                "seed": 0,
            })
            release_dir = outdir / "release_bundle"
            offline_dir = outdir / "offline_evaluation"
            manifest = json.loads((release_dir / "privacy_boundary_manifest.json").read_text())
            release_names = {path.name for path in release_dir.iterdir() if path.is_file()}
            self.assertEqual(set(manifest["allowed_files"]), release_names)
            self.assertIn("privacy_boundary_manifest.json", release_names)
            self.assertTrue((offline_dir / "record_assignments.npz").exists())
            self.assertFalse((release_dir / "record_assignments.npz").exists())

    def test_clipping_distortion_formula(self):
        U = np.asarray([[2.0, 0.0], [1.0, 2.0], [0.2, 0.3]])
        report = radial_l1_clipping_distortion(U, 1.0)
        self.assertTrue(report["formula_verified"])

    def test_matched_protocol_detects_mismatch(self):
        a = {"records": "x", "epsilon": 1, "L": 10}
        b = {"records": "x", "epsilon": 2, "L": 10}
        self.assertFalse(validate_matched_configs([a, b])["matched"])

    def test_paired_bootstrap_difference(self):
        report = paired_bootstrap_difference([1, 2, 3], [0, 1, 1], draws=200, seed=0)
        self.assertGreater(report["mean"], 0)
        self.assertEqual(report["n"], 3)

    def test_human_evaluation_blinding_and_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = [
                {"dataset": "demo", "target_sample_id": "s1", "method": "A", "summary": "alpha", "reference": "ref"},
                {"dataset": "demo", "target_sample_id": "s1", "method": "B", "summary": "beta", "reference": "ref"},
            ]
            tasks = root / "tasks.jsonl"
            key = root / "key.jsonl"
            create_blinded_batch(manifest, tasks, key, seed=3)
            task_rows = read_jsonl(tasks)
            key_rows = read_jsonl(key)
            self.assertNotIn("method", task_rows[0])
            annotations = []
            for task in task_rows:
                for annotator, score in [("r1", 4), ("r2", 5)]:
                    annotations.append({
                        "task_id": task["task_id"], "annotator_id": annotator,
                        "factual_support": score, "important_theme_coverage": score,
                        "unsupported_additions": 1, "usefulness": score, "clarity": score,
                    })
            report = evaluate_annotations(annotations, key_rows)
            self.assertEqual(report["design"]["n_tasks_retained"], 2)
            self.assertEqual(set(report["method_summary"]), {"A", "B"})

    def test_release_bundle_contains_no_offline_artifact_names(self):
        allowed = {
            "private_plan.json", "public_release_config.json", "structured_claims.json",
            "summary.txt", "verifier_output.json", "privacy_boundary_manifest.json",
        }
        release_dirs = list(Path("outputs/validation_demo").glob("*/release_bundle"))
        if release_dirs:
            names = {path.name for path in release_dirs[0].iterdir() if path.is_file()}
            self.assertTrue(names <= allowed)
            self.assertFalse({"clean_sketch.npy", "record_assignments.npz"} & names)


class ExternalValidityTests(unittest.TestCase):
    def test_two_embedding_families_are_fixed_and_bounded(self):
        records = [{"id": "r1", "text": "refund missing"}, {"id": "r2", "text": "login broken"}]
        atoms = [
            {"atom_id": 0, "label": "refund", "description": "refund not received"},
            {"atom_id": 1, "label": "account access", "description": "login access problem"},
        ]
        for family in ["tfidf_word", "tfidf_char"]:
            C, V, diagnostics = topK_sparse_assignments(
                records,
                atoms,
                K=1,
                embedding_family=family,
                return_diagnostics=True,
            )
            self.assertEqual(diagnostics["embedding"]["family"], family)
            self.assertTrue(np.all(np.asarray(V.sum(axis=1)).ravel() <= 1.0 + 1e-12))
            np.testing.assert_allclose(C, np.asarray(V.sum(axis=0)).ravel())

    def test_model_roles_reject_same_decoder_and_judge(self):
        from src.model_roles import validate_distinct_decoder_judge
        with self.assertRaises(ValueError):
            validate_distinct_decoder_judge(
                decoder_backend="openai",
                decoder_model="same-model",
                judge_backend="openai",
                judge_model="same-model",
                require_distinct=True,
            )
        report = validate_distinct_decoder_judge(
            decoder_backend="hf_local",
            decoder_model="open-model",
            judge_backend="openai",
            judge_model="judge-model",
            require_distinct=True,
        )
        self.assertTrue(report["valid"])

    def test_misspecification_stress_constructs_valid_dictionary(self):
        from src.external_validity import make_misspecified_dictionary
        atoms = [{"atom_id": i, "label": f"a{i}", "description": f"d{i}"} for i in range(10)]
        modified, report = make_misspecified_dictionary(
            atoms, fraction=0.3, mode="drop_random", seed=0
        )
        self.assertEqual(len(modified), 7)
        self.assertEqual(len(report["changed_atom_ids"]), 3)

    def test_scaling_benchmark_smoke(self):
        from src.scaling_benchmark import run_scaling_benchmark
        result = run_scaling_benchmark(
            {
                "m_values": [20],
                "L": 3,
                "mechanisms": ["gaussian", "private_topL"],
                "epsilon": 2.0,
                "delta": 1e-6,
                "warmup": 0,
                "repeats": 1,
                "seed": 0,
            }
        )
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(all(row["runtime_seconds_mean"] >= 0 for row in result["rows"]))

class EvaluationProtocolTests(unittest.TestCase):
    def test_assignment_bootstrap_and_paired_difference(self):
        from src.assignment_eval import bootstrap_assignment_metrics, paired_bootstrap_metric_difference
        annotations = [
            {"record_id": "1", "annotator_id": "a", "relevant_atom_ids": [0], "covered": True},
            {"record_id": "2", "annotator_id": "a", "relevant_atom_ids": [], "covered": False},
        ]
        good = [
            {"record_id": "1", "predicted_atom_ids": [0]},
            {"record_id": "2", "predicted_atom_ids": []},
        ]
        forced = [
            {"record_id": "1", "predicted_atom_ids": [0]},
            {"record_id": "2", "predicted_atom_ids": [0]},
        ]
        ci = bootstrap_assignment_metrics(good, annotations, n_bootstrap=50, seed=0)
        self.assertEqual(ci["n_records"], 2)
        diff = paired_bootstrap_metric_difference(
            good, forced, annotations, metric="coverage_rate", n_bootstrap=50, seed=0
        )
        self.assertLess(diff["estimate"], 0.0)

    def test_paired_transfer_bootstrap_accepts_dictionary_specific_annotations(self):
        from src.assignment_eval import paired_bootstrap_metric_difference_two_annotations
        annotations_a = [
            {"record_id": "1", "annotator_id": "a", "relevant_atom_ids": [0], "covered": True},
            {"record_id": "2", "annotator_id": "a", "relevant_atom_ids": [], "covered": False},
        ]
        annotations_b = [
            {"record_id": "1", "annotator_id": "a", "relevant_atom_ids": [10], "covered": True},
            {"record_id": "2", "annotator_id": "a", "relevant_atom_ids": [11], "covered": True},
        ]
        predictions_a = [
            {"record_id": "1", "predicted_atom_ids": [0]},
            {"record_id": "2", "predicted_atom_ids": []},
        ]
        predictions_b = [
            {"record_id": "1", "predicted_atom_ids": [10]},
            {"record_id": "2", "predicted_atom_ids": [11]},
        ]
        result = paired_bootstrap_metric_difference_two_annotations(
            predictions_a, annotations_a, predictions_b, annotations_b,
            metric="top1_precision", n_bootstrap=50, seed=0,
        )
        self.assertEqual(result["n_records"], 2)
        self.assertEqual(result["annotation_protocol"], "condition-specific annotations paired by target record_id")

    def test_sentence_encoder_is_registered_as_fixed_family(self):
        from src.embed import SUPPORTED_EMBEDDING_FAMILIES
        self.assertIn("sentence_transformer", SUPPORTED_EMBEDDING_FAMILIES)

    def test_scaling_benchmark_materializes_topL_payload(self):
        from src.scaling_benchmark import run_scaling_benchmark
        result = run_scaling_benchmark({
            "m_values": [20, 40], "L_values": [3], "mechanisms": ["gaussian", "private_topL"],
            "epsilon": 2.0, "delta": 1e-6, "warmup": 0, "repeats": 1, "seed": 9,
        })
        self.assertEqual(len(result["rows"]), 4)
        for row in result["rows"]:
            self.assertEqual(row["released_numeric_plan_bytes_mean"], 3 * 16)
        self.assertEqual(len(result["empirical_scaling_summary"]), 2)


if __name__ == "__main__":
    unittest.main()
