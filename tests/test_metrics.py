import numpy as np

from terrarium.core import metrics as M


def test_crps_point_forecast_is_absolute_error():
    assert np.isclose(M.crps_ensemble(np.array([[2.0]]), np.array([5.0]))[0], 3.0)


def test_crps_matches_closed_form_for_gaussian():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    # CRPS of N(0,1) at y=0 is (sqrt(2) - 1) / sqrt(pi)
    assert np.isclose(M.crps_ensemble(x, np.array(0.0)), (np.sqrt(2) - 1) / np.sqrt(np.pi), atol=0.01)


def test_calibrated_ensemble_has_uniform_pit_and_nominal_coverage():
    rng = np.random.default_rng(1)
    s = rng.normal(size=(5000, 8))
    y = rng.normal(size=5000)
    p = M.pit(s, y, rng)
    assert M.pit_ks(p) < 0.03
    assert abs(M.coverage_from_pit(p, 0.9).mean() - 0.9) < 0.02


def test_discrete_ties_do_not_look_miscalibrated():
    rng = np.random.default_rng(2)
    s = rng.integers(0, 3, size=(5000, 8)).astype(float)
    y = rng.integers(0, 3, size=5000).astype(float)
    assert M.pit_ks(M.pit(s, y, rng)) < 0.03


def test_point_forecast_has_zero_coverage():
    rng = np.random.default_rng(3)
    s = np.zeros((2000, 32))
    y = rng.normal(size=2000)
    assert M.coverage_from_pit(M.pit(s, y, rng), 0.9).mean() == 0.0
