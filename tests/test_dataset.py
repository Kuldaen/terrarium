import numpy as np

from terrarium.phenomena.erosion.cliff2d.dataset import EvalSet, PairDataset, generate_eval, generate_training
from terrarium.phenomena.erosion.cliff2d.summary import OUTCOMES, SUMMARY_FEATURES, cliff_outcomes


def test_training_pairs_roundtrip(tmp_path):
    generate_training(tmp_path / "tr", n_chains=4, years=1, shard_size=2, max_burn_in_days=0, device="cpu")
    ds = PairDataset(tmp_path / "tr", dts=(30, 90))
    assert len(ds) > 0
    b = ds.sample(8, np.random.default_rng(0))
    assert b["summary"].shape == (8, len(SUMMARY_FEATURES))
    assert b["outcomes"].shape == (8, len(OUTCOMES))
    assert (b["outcomes"][:, 2] >= 0).all()  # volume lost is non-negative


def test_eval_ensembles_shapes_and_consistency(tmp_path):
    generate_eval(tmp_path / "ev", n_starts=2, n_futures=3, burn_in_days=30, dts=(30, 90), device="cpu")
    ev = EvalSet(tmp_path / "ev")
    assert ev.outcomes.shape == (2, 3, 2, len(OUTCOMES))
    recomputed = cliff_outcomes(ev.start_rock[:, None, None], ev.rock, ev.top_row[:, None, None])
    np.testing.assert_allclose(recomputed, ev.outcomes, atol=1e-3)
