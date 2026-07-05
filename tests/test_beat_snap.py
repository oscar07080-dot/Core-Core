from coreedit.audio import snap_to_grid


def test_snaps_to_nearest_within_tolerance():
    grid = [0.0, 0.5, 1.0, 1.5]
    assert snap_to_grid(0.55, grid, tolerance=0.1) == 0.5
    assert snap_to_grid(0.95, grid, tolerance=0.1) == 1.0


def test_exact_match_returns_candidate():
    assert snap_to_grid(1.0, [0.0, 1.0, 2.0], tolerance=0.01) == 1.0


def test_outside_tolerance_returns_value():
    assert snap_to_grid(0.75, [0.0, 0.5, 1.0], tolerance=0.1) == 0.75


def test_before_first_and_after_last():
    grid = [1.0, 2.0]
    assert snap_to_grid(0.95, grid, tolerance=0.1) == 1.0
    assert snap_to_grid(0.5, grid, tolerance=0.1) == 0.5
    assert snap_to_grid(2.05, grid, tolerance=0.1) == 2.0
    assert snap_to_grid(3.0, grid, tolerance=0.1) == 3.0


def test_empty_candidates():
    assert snap_to_grid(1.23, [], tolerance=1.0) == 1.23


def test_ties_pick_a_candidate():
    # equidistant between 0.5 and 1.0; either is acceptable, must be snapped
    result = snap_to_grid(0.75, [0.5, 1.0], tolerance=0.3)
    assert result in (0.5, 1.0)
