"""
Benchmark *geometry* sanity checks — no FEM solves.

These tests catch typos in node coordinates, connectivity lists, group
maps, and unit declarations. They run in milliseconds so they're cheap
to run on every commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.registry import available_benchmarks, get_benchmark


VERIFIED = ["10bar", "25bar", "72bar", "200bar"]


def test_registry_lists_four_benchmarks():
    names = available_benchmarks()
    assert set(names) == {"10bar", "25bar", "72bar", "200bar"}


def test_every_benchmark_encoded_after_phase7():
    # All four benchmarks should construct without raising after Phase 7.
    for name in VERIFIED:
        b = get_benchmark(name)
        assert b.reference_verified, f"{name} must be flagged as verified"


@pytest.mark.parametrize("name", VERIFIED)
def test_nodes_and_connectivity_shapes(name):
    b = get_benchmark(name)
    assert b.nodes.ndim == 2
    assert b.nodes.shape[1] in (2, 3)
    assert b.connectivity.shape[1] == 2
    # connectivity must reference only existing nodes.
    assert b.connectivity.min() >= 0
    assert b.connectivity.max() < b.nodes.shape[0]
    # no self-loops.
    assert (b.connectivity[:, 0] != b.connectivity[:, 1]).all()


@pytest.mark.parametrize("name", VERIFIED)
def test_group_map_partitions_elements(name):
    b = get_benchmark(name)
    all_idx = [i for grp in b.group_map for i in grp]
    assert sorted(all_idx) == list(range(b.n_bars))


@pytest.mark.parametrize("name", VERIFIED)
def test_area_bounds_and_material_positive(name):
    b = get_benchmark(name)
    lo, hi = b.area_bounds
    assert 0 < lo < hi
    assert b.E > 0
    assert b.density > 0
    assert b.stress_limit_tension > 0
    assert b.stress_limit_compression > 0
    assert b.displacement_limit > 0


@pytest.mark.parametrize("name", VERIFIED)
def test_bar_lengths_positive(name):
    b = get_benchmark(name)
    assert (b._bar_lengths > 0).all()
    assert np.isfinite(b._bar_lengths).all()


def test_10bar_specific_geometry():
    """Quick spot-checks unique to Sunar-Belegundu 10-bar."""
    b = get_benchmark("10bar")
    assert b.n_bars == 10
    assert b.n_design_vars == 10
    assert b.ndim == 2
    assert b.units == "imperial"
    # Six orthogonal bars should all be 360 in long; four diagonals sqrt(2)*360.
    ortho = b._bar_lengths[:6]
    diag = b._bar_lengths[6:]
    np.testing.assert_allclose(ortho, 360.0)
    np.testing.assert_allclose(diag, 360.0 * np.sqrt(2.0))


def test_25bar_specific_geometry():
    b = get_benchmark("25bar")
    assert b.n_bars == 25
    assert b.n_design_vars == 8
    assert b.ndim == 3
    assert b.units == "imperial"
    # Four pinned supports at the base.
    assert len(b.supports) == 4
    # Two load cases.
    assert len(b.load_cases) == 2


def test_72bar_specific_geometry():
    b = get_benchmark("72bar")
    assert b.n_bars == 72
    assert b.n_design_vars == 16
    assert b.ndim == 3
    assert b.units == "imperial"
    assert len(b.supports) == 4
    assert len(b.load_cases) == 2
    # Every story contributes 18 bars (4+8+4+2) for 4 stories -> 72 total.
    assert sum(len(g) for g in b.group_map) == 72


def test_200bar_specific_geometry():
    b = get_benchmark("200bar")
    assert b.n_bars == 200
    assert b.n_design_vars == 29
    assert b.ndim == 2
    assert b.units == "SI"
    # Seven pinned base supports (one per bottom-row column).
    assert len(b.supports) == 7
    # Three load cases (lateral, gravity, combined).
    assert len(b.load_cases) == 3
    # Steel material sanity.
    assert b.E >= 2e11
    assert b.density >= 7000
