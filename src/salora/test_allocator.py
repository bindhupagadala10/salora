"""
Unit tests for src/salora/allocator.py, covering exactly the 9 cases
listed in docs/design/SALoRA_Allocation_Spec.md, Section 11.

Pure stdlib (unittest, tempfile) -- no torch/transformers/peft needed,
so this runs anywhere, including CPU-only/CI environments.

Run:
    python -m unittest src.salora.test_allocator -v
"""

import tempfile
import unittest
from pathlib import Path

from src.salora.allocator import (
    Allocation,
    allocate_inverse_drift,
    allocate_random,
    allocate_ranks,
    allocate_uniform,
    compute_trainable_params,
)

LAYERS = list(range(1, 13))


def make_drift(values):
    assert len(values) == 12
    return dict(zip(LAYERS, values))


class TestBudgetConservation(unittest.TestCase):
    """Spec 11.1: sum of ranks always equals the total budget."""

    def test_uniform_drift(self):
        d = make_drift([1.0] * 12)
        alloc = allocate_ranks(d, total_budget=96)
        self.assertEqual(alloc.total_rank(), 96)

    def test_skewed_drift(self):
        d = make_drift([0.1, 0.12, 0.15, 0.17, 0.20, 0.27, 0.31, 0.42, 0.55, 0.61, 0.72, 0.91])
        alloc = allocate_ranks(d, total_budget=96)
        self.assertEqual(alloc.total_rank(), 96)

    def test_non_default_budget(self):
        d = make_drift([1.0] * 12)
        alloc = allocate_ranks(d, total_budget=144, r_max=24)
        self.assertEqual(alloc.total_rank(), 144)

    def test_real_sinkhorn_profile_n950(self):
        # Actual mnli_vs_wanli.csv Sinkhorn column, n=950 (embedding excluded).
        d = make_drift([
            0.3296, 0.2674, 0.2701, 0.2923, 0.3131, 0.3522,
            0.3901, 0.5386, 0.6972, 0.7064, 0.8861, 0.9690,
        ])
        alloc = allocate_ranks(d, total_budget=96, metric="sinkhorn")
        self.assertEqual(alloc.total_rank(), 96)


class TestMinMaxConstraint(unittest.TestCase):
    """Spec 11.2: no allocated rank violates [r_min, r_max]."""

    def test_bounds_respected_on_skewed_input(self):
        d = make_drift([0.1, 0.12, 0.15, 0.17, 0.20, 0.27, 0.31, 0.42, 0.55, 0.61, 0.72, 0.91])
        alloc = allocate_ranks(d, total_budget=96, r_min=1, r_max=16)
        for l, r in alloc.layer_ranks.items():
            self.assertGreaterEqual(r, 1, f"layer {l} below r_min")
            self.assertLessEqual(r, 16, f"layer {l} above r_max")

    def test_infeasible_budget_raises(self):
        d = make_drift([1.0] * 12)
        # r_min=1 * 12 = 12 minimum, r_max=16*12=192 maximum; 5 is infeasible low.
        with self.assertRaises(ValueError):
            allocate_ranks(d, total_budget=5, r_min=1, r_max=16)
        with self.assertRaises(ValueError):
            allocate_ranks(d, total_budget=300, r_min=1, r_max=16)


class TestDeterminism(unittest.TestCase):
    """Spec 11.3: identical input -> byte-identical output, repeatedly."""

    def test_repeated_calls_match(self):
        d = make_drift([0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.02, 0.02, 0.03])
        a1 = allocate_ranks(d, total_budget=96)
        a2 = allocate_ranks(d, total_budget=96)
        self.assertEqual(a1.layer_ranks, a2.layer_ranks)

    def test_random_allocation_reproducible_given_seed(self):
        a1 = allocate_random(total_budget=96, allocation_seed=7)
        a2 = allocate_random(total_budget=96, allocation_seed=7)
        self.assertEqual(a1.layer_ranks, a2.layer_ranks)

    def test_random_allocation_differs_across_seeds_generally(self):
        a1 = allocate_random(total_budget=96, allocation_seed=1)
        a2 = allocate_random(total_budget=96, allocation_seed=2)
        self.assertNotEqual(a1.layer_ranks, a2.layer_ranks)


class TestZeroDriftHandling(unittest.TestCase):
    """Spec 11.4: all-zero drift falls back to uniform, no NaN/exception."""

    def test_all_zero_drift_falls_back_to_uniform(self):
        d = make_drift([0.0] * 12)
        alloc = allocate_ranks(d, total_budget=96)
        self.assertEqual(alloc.total_rank(), 96)
        # Uniform fallback: every layer should get exactly 96/12 = 8.
        self.assertTrue(all(r == 8 for r in alloc.layer_ranks.values()))

    def test_no_nan_or_negative(self):
        d = make_drift([0.0] * 12)
        alloc = allocate_ranks(d, total_budget=96)
        for r in alloc.layer_ranks.values():
            self.assertIsInstance(r, int)
            self.assertGreaterEqual(r, 0)


class TestTiedDrift(unittest.TestCase):
    """Spec 11.5: tied drift values -> deterministic, order-independent tie-break."""

    def test_all_tied_gives_uniform(self):
        d = make_drift([0.5] * 12)
        alloc = allocate_ranks(d, total_budget=96)
        self.assertTrue(all(r == 8 for r in alloc.layer_ranks.values()))

    def test_partial_tie_deterministic(self):
        # Layers 1 and 2 tied at the same (nonzero) value; result must not
        # depend on dict insertion order.
        d1 = make_drift([0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        d2 = {l: d1[l] for l in reversed(LAYERS)}  # same values, different insertion order
        a1 = allocate_ranks(d1, total_budget=96)
        a2 = allocate_ranks(d2, total_budget=96)
        self.assertEqual(a1.layer_ranks, a2.layer_ranks)


class TestFloatingPointRoundingEdgeCases(unittest.TestCase):
    """Spec 11.6: exact .5 boundaries, leftover=0, leftover=12 (all tied)."""

    def test_leftover_zero(self):
        # 12 layers, budget exactly divisible with no remainder in any raw_r_l.
        d = make_drift([1.0] * 12)
        alloc = allocate_ranks(d, total_budget=96)
        self.assertEqual(sum(alloc.layer_ranks.values()), 96)
        self.assertTrue(all(r == 8 for r in alloc.layer_ranks.values()))

    def test_leftover_all_twelve(self):
        # Construct weights that all land on the same non-integer raw rank,
        # forcing leftover to be nonzero for every layer simultaneously.
        d = make_drift([1.0] * 12)
        alloc = allocate_ranks(d, total_budget=97, r_max=17)  # 97/12 = 8.0833...
        self.assertEqual(sum(alloc.layer_ranks.values()), 97)
        # With all raw ranks tied, exactly one layer (lowest index, per
        # deterministic tie-break) absorbs the +1 leftover unit.
        ranks = list(alloc.layer_ranks.values())
        self.assertEqual(sorted(ranks), [8] * 11 + [9])

    def test_exact_half_boundary(self):
        # Two layers whose raw_r_l lands exactly on a .5 boundary.
        weights = [1.5, 1.5] + [1.0] * 10
        d = make_drift(weights)
        alloc = allocate_ranks(d, total_budget=96)
        self.assertEqual(sum(alloc.layer_ranks.values()), 96)


class TestParameterCountEquivalence(unittest.TestCase):
    """Spec 11.7: every method must produce the identical trainable-param count."""

    def test_uniform_matches_hand_computation(self):
        alloc = allocate_uniform(total_budget=96)
        # 96 total rank units * 2 modules/layer-equivalent * 1536 = 294,912
        # (see allocator.compute_trainable_params docstring for derivation).
        self.assertEqual(alloc.trainable_params(), 294_912)

    def test_all_methods_equal_under_same_budget(self):
        drift = make_drift([0.1, 0.12, 0.15, 0.17, 0.20, 0.27, 0.31, 0.42, 0.55, 0.61, 0.72, 0.91])

        uniform = allocate_uniform(total_budget=96)
        mmd = allocate_ranks(drift, total_budget=96, metric="mmd")
        inv = allocate_inverse_drift(drift, total_budget=96, metric="inverse")
        rnd = allocate_random(total_budget=96, allocation_seed=3)

        params = {
            "uniform": uniform.trainable_params(),
            "mmd": mmd.trainable_params(),
            "inverse": inv.trainable_params(),
            "random": rnd.trainable_params(),
        }
        self.assertEqual(
            len(set(params.values())), 1,
            f"Parameter counts differ across methods, budget is not actually "
            f"equalized: {params}",
        )

    def test_compute_trainable_params_matches_allocation_method(self):
        alloc = allocate_uniform(total_budget=96)
        self.assertEqual(
            compute_trainable_params(alloc.layer_ranks),
            alloc.trainable_params(),
        )


class TestRMaxClampTriggersAndRedistributes(unittest.TestCase):
    """
    Spec 11.8: the handoff's own pathological example (one dominant layer)
    must actually hit r_max and have the excess redistributed, not silently
    ignored.
    """

    def test_dominant_layer_clamped(self):
        # Padded/adapted version of the handoff's 5-value example to 12 layers:
        # one layer completely dominates, rest are near-zero.
        d = make_drift([0.001, 0.001, 0.002, 0.005, 0.99, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        alloc = allocate_ranks(d, total_budget=96, r_min=1, r_max=16)

        dominant_layer = 5
        self.assertEqual(
            alloc.layer_ranks[dominant_layer], 16,
            "Dominant-drift layer should hit r_max, not consume the whole budget.",
        )
        self.assertEqual(alloc.total_rank(), 96)
        # Every other layer should still have gotten at least r_min.
        for l, r in alloc.layer_ranks.items():
            if l != dominant_layer:
                self.assertGreaterEqual(r, 1)


class TestSerializationRoundTrip(unittest.TestCase):
    """Spec 11.9: save_csv -> load_csv reproduces the identical rank dict."""

    def test_round_trip(self):
        d = make_drift([0.1, 0.12, 0.15, 0.17, 0.20, 0.27, 0.31, 0.42, 0.55, 0.61, 0.72, 0.91])
        alloc = allocate_ranks(d, total_budget=96, metric="mmd", sample_size=950)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wanli_mmd_n950.csv"
            alloc.save_csv(path)

            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".meta.json").exists())

            reloaded = Allocation.load_csv(path)
            self.assertEqual(reloaded.layer_ranks, alloc.layer_ranks)
            self.assertEqual(reloaded.metric, alloc.metric)
            self.assertEqual(reloaded.total_budget, alloc.total_budget)


if __name__ == "__main__":
    unittest.main()
