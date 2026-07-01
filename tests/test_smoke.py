"""Smoke tests for the core DP-SPIN release and plan-construction functions."""

from __future__ import annotations

import unittest

import numpy as np

from src.dp_mechanisms import gaussian_release, laplace_release, topL
from src.plan import make_private_plan


class CoreReleaseSmokeTests(unittest.TestCase):
    """Check the basic contracts of the core release functions."""

    def setUp(self) -> None:
        self.sketch = np.array([10.0, 5.0, 1.0])
        self.atoms = [
            {"atom_id": index, "label": f"a{index}", "description": ""}
            for index in range(self.sketch.size)
        ]

    def test_topL_returns_largest_indices_with_deterministic_ties(self) -> None:
        values = np.array([3.0, 5.0, 5.0, 1.0])
        np.testing.assert_array_equal(topL(values, 2), np.array([1, 2]))

    def test_gaussian_release_preserves_shape_and_records_metadata(self) -> None:
        released, metadata = gaussian_release(
            self.sketch,
            epsilon=1.0,
            delta=1e-6,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(released.shape, self.sketch.shape)
        self.assertTrue(np.all(np.isfinite(released)))
        self.assertEqual(metadata["mechanism"], "gaussian")
        self.assertEqual(metadata["epsilon"], 1.0)
        self.assertEqual(metadata["delta"], 1e-6)
        self.assertGreater(metadata["sigma"], 0.0)

    def test_laplace_release_preserves_shape_and_records_metadata(self) -> None:
        released, metadata = laplace_release(
            self.sketch,
            epsilon=1.0,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(released.shape, self.sketch.shape)
        self.assertTrue(np.all(np.isfinite(released)))
        self.assertEqual(metadata["mechanism"], "laplace")
        self.assertEqual(metadata["epsilon"], 1.0)
        self.assertEqual(metadata["delta"], 0.0)
        self.assertGreater(metadata["scale"], 0.0)

    def test_private_plan_contains_only_the_requested_number_of_atoms(self) -> None:
        plan = make_private_plan(
            self.sketch,
            self.atoms,
            L=2,
            public_denominator=100,
        )

        self.assertEqual(plan["L"], 2)
        self.assertEqual(plan["public_denominator"], 100)
        self.assertEqual(len(plan["selected_atoms"]), 2)
        self.assertEqual(
            [atom["atom_id"] for atom in plan["selected_atoms"]],
            [0, 1],
        )


if __name__ == "__main__":
    unittest.main()
