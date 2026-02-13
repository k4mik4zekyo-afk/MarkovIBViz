"""
tests_structural.py — Validate market structure assumptions.

Per Agent.md, Structural Tests (Research Validity):
- Acceptance probability vs failure count behaves monotonically
- Rejection probability increases after repeated failures
- Extension distributions widen with persistence
- Upside and downside statistics are approximately symmetric
- Sample sizes per state exceed minimum thresholds

Tests are run separately for pre-IB and post-IB phases.

Failure here indicates invalid assumptions or insufficient data, not code errors.
"""

import pandas as pd
import numpy as np
import os
import sys

from episode_logger import build_episode_log
from probability_engine import compute_probabilities

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
REPORT_FILE = os.path.join(OUTPUT_DIR, "structural_test_report.txt")

MIN_SAMPLE_SIZE = 10
SYMMETRY_TOLERANCE = 3.0


class StructuralTestRunner:
    """Runs structural tests on episode data for a given phase."""

    def __init__(self, ep_df, prob_table, ext_stats, phase_label):
        self.ep_df = ep_df
        self.prob_table = prob_table
        self.ext_stats = ext_stats
        self.phase_label = phase_label
        self.results = []

    def run_test(self, name, func):
        """Run a single test and record result."""
        try:
            passed, detail = func()
            self.results.append({
                "test": f"[{self.phase_label}] {name}",
                "passed": passed,
                "detail": detail,
            })
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] [{self.phase_label}] {name}: {detail}")
        except Exception as e:
            self.results.append({
                "test": f"[{self.phase_label}] {name}",
                "passed": False,
                "detail": f"Exception: {e}",
            })
            print(f"  [FAIL] [{self.phase_label}] {name}: Exception: {e}")

    def test_acceptance_monotonic(self):
        """Acceptance probability vs failure count behaves monotonically."""
        if self.prob_table.empty:
            return False, "No probability data"

        results_by_direction = {}
        for direction in ["up", "down"]:
            mask = (self.prob_table["direction"] == direction) & (self.prob_table["outcome"] == "A")
            subset = self.prob_table[mask].copy()
            if subset.empty:
                continue

            subset["fc"] = subset["discovery_state"].str.replace("D", "").astype(int)
            subset = subset.sort_values("fc")
            subset = subset[subset["total"] >= MIN_SAMPLE_SIZE]

            if len(subset) < 2:
                results_by_direction[direction] = "insufficient_states"
                continue

            probs = subset["probability"].values
            violations = sum(1 for i in range(len(probs) - 1) if probs[i + 1] > probs[i])
            max_violations = max(1, len(probs) // 3)
            results_by_direction[direction] = violations <= max_violations

        if not results_by_direction:
            return False, "No directions with sufficient data"

        checked = {d: v for d, v in results_by_direction.items() if v != "insufficient_states"}
        if not checked:
            return False, "Insufficient states with enough samples in all directions"

        all_pass = all(v is True for v in checked.values())
        detail = "; ".join(f"{d}={'pass' if v is True else 'fail'}" for d, v in results_by_direction.items())
        return all_pass, f"Acceptance monotonicity: {detail}"

    def test_rejection_increases(self):
        """Rejection probability increases after repeated failures."""
        if self.prob_table.empty:
            return False, "No probability data"

        results_by_direction = {}
        for direction in ["up", "down"]:
            mask = (self.prob_table["direction"] == direction) & (self.prob_table["outcome"] == "R")
            subset = self.prob_table[mask].copy()
            if subset.empty:
                continue

            subset["fc"] = subset["discovery_state"].str.replace("D", "").astype(int)
            subset = subset.sort_values("fc")
            subset = subset[subset["total"] >= MIN_SAMPLE_SIZE]

            if len(subset) < 2:
                results_by_direction[direction] = "insufficient_states"
                continue

            probs = subset["probability"].values
            violations = sum(1 for i in range(len(probs) - 1) if probs[i + 1] < probs[i])
            max_violations = max(1, len(probs) // 3)
            results_by_direction[direction] = violations <= max_violations

        if not results_by_direction:
            return False, "No directions with sufficient data"

        checked = {d: v for d, v in results_by_direction.items() if v != "insufficient_states"}
        if not checked:
            return False, "Insufficient states with enough samples"

        all_pass = all(v is True for v in checked.values())
        detail = "; ".join(f"{d}={'pass' if v is True else 'fail'}" for d, v in results_by_direction.items())
        return all_pass, f"Rejection increase: {detail}"

    def test_extension_widening(self):
        """Extension distributions widen with persistence.

        For post-IB phase, uses IB-normalized extensions if available.
        """
        if self.ext_stats.empty:
            return False, "No extension statistics"

        # Use IB-normalized std if available (post-IB), else raw
        std_col = "std_extension_ib" if "std_extension_ib" in self.ext_stats.columns else "std_extension"

        results_by_direction = {}
        for direction in ["up", "down"]:
            subset = self.ext_stats[self.ext_stats["direction"] == direction].copy()
            if subset.empty:
                continue

            subset["fc"] = subset["discovery_state"].str.replace("D", "").astype(int)
            subset = subset.sort_values("fc")
            subset = subset[subset["n"] >= MIN_SAMPLE_SIZE]

            if len(subset) < 2:
                results_by_direction[direction] = "insufficient_states"
                continue

            stds = subset[std_col].values
            violations = sum(1 for i in range(len(stds) - 1) if stds[i + 1] < stds[i])
            max_violations = max(1, len(stds) // 3)
            results_by_direction[direction] = violations <= max_violations

        if not results_by_direction:
            return False, "No directions with sufficient data"

        checked = {d: v for d, v in results_by_direction.items() if v != "insufficient_states"}
        if not checked:
            return False, "Insufficient states with enough samples"

        all_pass = all(v is True for v in checked.values())
        detail = "; ".join(f"{d}={'pass' if v is True else 'fail'}" for d, v in results_by_direction.items())
        return all_pass, f"Extension widening ({std_col}): {detail}"

    def test_symmetry(self):
        """Upside and downside statistics are approximately symmetric."""
        if self.ep_df.empty:
            return False, "No episodes"

        up = self.ep_df[self.ep_df["direction"] == "up"]
        down = self.ep_df[self.ep_df["direction"] == "down"]

        if len(up) == 0 or len(down) == 0:
            return False, "Missing one direction entirely"

        count_ratio = max(len(up), len(down)) / min(len(up), len(down))

        up_ext = up["max_extension"].mean()
        down_ext = down["max_extension"].mean()
        if min(up_ext, down_ext) > 0:
            ext_ratio = max(up_ext, down_ext) / min(up_ext, down_ext)
        else:
            ext_ratio = float("inf")

        count_ok = count_ratio <= SYMMETRY_TOLERANCE
        ext_ok = ext_ratio <= SYMMETRY_TOLERANCE

        passed = count_ok and ext_ok
        detail = (
            f"count_ratio={count_ratio:.2f} ({'ok' if count_ok else 'skewed'}), "
            f"ext_ratio={ext_ratio:.2f} ({'ok' if ext_ok else 'skewed'}), "
            f"up_n={len(up)}, down_n={len(down)}"
        )
        return passed, detail

    def test_sample_sizes(self):
        """Sample sizes per state exceed minimum thresholds."""
        if self.ep_df.empty:
            return False, "No episodes"

        counts = self.ep_df.groupby(["direction", "discovery_state"]).size().reset_index(name="n")

        insufficient = []
        for direction in ["up", "down"]:
            d0 = counts[(counts["direction"] == direction) & (counts["discovery_state"] == "D0")]
            if d0.empty or d0["n"].iloc[0] < MIN_SAMPLE_SIZE:
                n = d0["n"].iloc[0] if not d0.empty else 0
                insufficient.append(f"{direction}/D0 (n={n})")

        passed = len(insufficient) == 0
        detail = f"Insufficient: {insufficient}" if insufficient else (
            f"All base states have >= {MIN_SAMPLE_SIZE} samples"
        )
        return passed, detail

    def run_all(self):
        """Run all structural tests for this phase."""
        self.run_test(
            "Acceptance probability monotonically decreasing",
            self.test_acceptance_monotonic,
        )
        self.run_test(
            "Rejection probability increases with failures",
            self.test_rejection_increases,
        )
        self.run_test(
            "Extension distributions widen with persistence",
            self.test_extension_widening,
        )
        self.run_test(
            "Upside/downside approximately symmetric",
            self.test_symmetry,
        )
        self.run_test(
            "Sample sizes exceed minimum thresholds",
            self.test_sample_sizes,
        )

        return self.results


def run_all_structural_tests():
    """Run structural tests for both pre-IB and post-IB phases."""
    print("=" * 60)
    print("STRUCTURAL TESTS")
    print("=" * 60)

    pre_ib_df, post_ib_df, _, _ = build_episode_log()

    all_results = []

    # Pre-IB structural tests
    if not pre_ib_df.empty:
        print(f"\n  --- Pre-IB Phase ({len(pre_ib_df)} episodes) ---")
        pre_prob, _, pre_ext = compute_probabilities(pre_ib_df)
        pre_runner = StructuralTestRunner(pre_ib_df, pre_prob, pre_ext, "Pre-IB")
        all_results.extend(pre_runner.run_all())
    else:
        print("  No pre-IB episodes. Skipping pre-IB structural tests.")

    # Post-IB structural tests
    if not post_ib_df.empty:
        print(f"\n  --- Post-IB Phase ({len(post_ib_df)} episodes) ---")
        post_prob, _, post_ext = compute_probabilities(post_ib_df)
        post_runner = StructuralTestRunner(post_ib_df, post_prob, post_ext, "Post-IB")
        all_results.extend(post_runner.run_all())
    else:
        print("  No post-IB episodes. Skipping post-IB structural tests.")

    n_passed = sum(1 for r in all_results if r["passed"])
    n_total = len(all_results)

    print("=" * 60)
    print(f"STRUCTURAL TESTS: {n_passed}/{n_total} passed")
    if n_passed == n_total:
        print("STATUS: ALL STRUCTURAL TESTS PASSED")
    else:
        print("STATUS: SOME STRUCTURAL TESTS FAILED")
        print("(Structural test failures indicate invalid assumptions")
        print(" or insufficient data, not code errors.)")
        for r in all_results:
            if not r["passed"]:
                print(f"  FAILED: {r['test']} — {r['detail']}")
    print("=" * 60)

    return n_passed == n_total, all_results


def save_structural_report(results, output_path=None):
    """Save structural test report."""
    if output_path is None:
        output_path = REPORT_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = ["STRUCTURAL TEST REPORT", "=" * 40]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(f"[{status}] {r['test']}: {r['detail']}")

    n_passed = sum(1 for r in results if r["passed"])
    lines.append("=" * 40)
    lines.append(f"Result: {n_passed}/{len(results)} passed")
    lines.append("")
    lines.append("Note: Structural test failures indicate invalid assumptions")
    lines.append("or insufficient data, not code errors.")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved structural test report to {output_path}")


if __name__ == "__main__":
    all_passed, results = run_all_structural_tests()
    save_structural_report(results)
