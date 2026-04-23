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


VERIFIED = ["10bar"]  # Benchmarks with fully-encoded geometry in Phase 1.


def test_registry_lists_four_benchmarks():
    names = available_benchmarks()
    assert set(names) == {"10bar", "25bar", "72bar", "200bar"}


def test_unverified_benchmarks_raise_not_implemented():
    for name in ("25bar", "72bar", "200bar"):
        with pytest.raises(NotImplementedError):
            get_benchmark(name)


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
