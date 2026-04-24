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


VERIFIED = ["10bar", "25bar", "72bar", "200bar"]  # fully-encoded geometries


def test_registry_lists_four_benchmarks():
    names = available_benchmarks()
    assert set(names) == {"10bar", "25bar", "72bar", "200bar"}


def test_unverified_benchmarks_raise_not_implemented():
    # All four benchmarks now resolve; no NotImplementedError expected.
    for name in ("10bar", "25bar", "72bar", "200bar"):
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


def test_25bar_specific_geometry():
    """Quick spot-checks unique to Schmit-Miura 25-bar."""
    b = get_benchmark("25bar")
    assert b.n_bars == 25
    assert b.n_design_vars == 8
    assert b.ndim == 3
    assert b.units == "imperial"
    # Group sizes must match the 1+4+4+2+2+4+4+4 literature grouping.
    assert [len(g) for g in b.group_map] == [1, 4, 4, 2, 2, 4, 4, 4]
    # Top-chord bar (index 0) is the only 75-in bar in group 0.
    np.testing.assert_allclose(b._bar_lengths[0], 75.0)
    # Base is at +/- 100 in x and y; support nodes are 6..9.
    np.testing.assert_allclose(
        np.abs(b.nodes[6:10, :2]),
        np.full((4, 2), 100.0),
    )


def test_72bar_specific_geometry():
    """Quick spot-checks unique to Fleury-Schmit 72-bar tower."""
    b = get_benchmark("72bar")
    assert b.n_bars == 72
    assert b.n_design_vars == 16
    assert b.ndim == 3
    assert b.units == "imperial"
    # Four storeys of [columns(4), face_diag(8), top_horiz(4), plan_diag(2)] —
    # ordering matches Schmit-Farshi 1974 / Camp 2007 so published optima can
    # be plugged in directly.
    assert [len(g) for g in b.group_map] == [4, 8, 4, 2] * 4
    # Base nodes 16..19 at z=0, tip nodes 12..15 at z=240.
    np.testing.assert_allclose(b.nodes[16:20, 2], 0.0)
    np.testing.assert_allclose(b.nodes[12:16, 2], 240.0)
    # Shortest bar type is the 60-in column between storey layers.
    assert abs(b._bar_lengths.min() - 60.0) < 1e-6
    # Displacement constraint applies only to lateral DOFs of tip nodes.
    assert b.displacement_check_dofs is not None
    assert set(b.displacement_check_dofs) == {(n, ax) for n in (12, 13, 14, 15) for ax in (0, 1)}


def test_200bar_specific_geometry():
    """Kaveh-style stepped-tower 200-bar (SI, steel)."""
    b = get_benchmark("200bar")
    assert b.n_bars == 200
    assert b.n_design_vars == 29
    assert b.ndim == 2
    assert b.units == "SI"
    # Steel properties.
    assert abs(b.E - 2.10e11) < 1e3
    assert abs(b.density - 7850.0) < 1e-6
    # Three load cases.
    assert len(b.load_cases) == 3
    # 11 levels x 7 columns = 77 nodes.
    assert b.nodes.shape[0] == 77
    # Base level (y=0) nodes 0..6, top level (y=30) nodes 70..76.
    np.testing.assert_allclose(b.nodes[0:7, 1], 0.0)
    np.testing.assert_allclose(b.nodes[70:77, 1], 30.0)
    # Group sizes: 10 columns of 7 bars each, 11 horizontal rows of 6 bars
    # each, plus 8 diagonal groups totalling 64 bars.
    col_sizes = [len(g) for g in b.group_map[:10]]
    horiz_sizes = [len(g) for g in b.group_map[10:21]]
    diag_sizes = [len(g) for g in b.group_map[21:29]]
    assert col_sizes == [7] * 10
    assert horiz_sizes == [6] * 11
    assert sum(diag_sizes) == 64  # 60 single-diags + 4 extra X-braces
    # Area bounds match Kaveh & Talatahari 2010 spec.
    assert abs(b.area_bounds[0] - 6.5e-5) < 1e-10
    assert abs(b.area_bounds[1] - 2.5e-2) < 1e-10
    # Displacement check restricted to top-level vertical DOFs.
    assert b.displacement_check_dofs is not None
    expected = {(n, 1) for n in range(70, 77)}
    assert set(b.displacement_check_dofs) == expected
