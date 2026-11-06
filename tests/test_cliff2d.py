import numpy as np
import pytest

from terrarium.phenomena.erosion.cliff2d import Cliff2D, generate_cliffs, generate_forcing

B, DAYS = 8, 365


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(42)
    init = generate_cliffs(B, rng)
    forcing = generate_forcing(B, DAYS, rng)
    return init, forcing


def run(init, forcing, seeds, crest=None, days=None):
    sim = Cliff2D(init, device="cpu")
    if crest is not None:
        sim.set_actions(revetment_crest=crest)
    return sim, sim.run(forcing, seeds=seeds, snapshot_every=DAYS, days=days)


def test_deterministic_given_forcing_and_seeds(setup):
    init, forcing = setup
    _, a = run(init, forcing, np.arange(B))
    _, b = run(init, forcing, np.arange(B))
    np.testing.assert_array_equal(a.snapshots["rock"], b.snapshots["rock"])


def test_rock_only_decreases(setup):
    init, forcing = setup
    _, r = run(init, forcing, np.arange(B))
    s = r.snapshots["rock"].astype(np.float32)
    assert (s[:, -1] <= s[:, 0] + 1e-3).all()
    assert (s[:, 0] - s[:, -1]).sum() > 0  # something eroded


def test_branching_matches_straight_run(setup):
    """Run 365 days straight vs 180 + 185 from a saved state."""
    init, forcing = setup
    _, full = run(init, forcing, np.arange(B))
    sim, _ = run(init, forcing, np.arange(B), days=180)
    state = sim.get_state()
    sim2 = Cliff2D(init, device="cpu")
    sim2.set_state(state)
    rest = sim2.run(forcing, seeds=np.arange(B), snapshot_every=185, days=185)
    np.testing.assert_array_equal(full.snapshots["rock"][:, -1], rest.snapshots["rock"][:, -1])


def test_revetment_reduces_toe_erosion(setup):
    init, forcing = setup
    _, none = run(init, forcing, np.arange(B))
    _, rev = run(init, forcing, np.arange(B), crest=np.full(B, 3.0, np.float32))
    lost = lambda r: (r.snapshots["rock"][:, 0].astype(np.float32) - r.snapshots["rock"][:, -1].astype(np.float32))
    # rows below +3 m: index (3 - z0)/dz = 28
    assert lost(rev)[:, :28].sum() < 0.5 * lost(none)[:, :28].sum()


def test_onshore_wind_increases_upper_face_weathering(setup):
    init, forcing = setup
    calm = dict(forcing, wind=np.zeros_like(forcing["hs"]))
    windy = dict(forcing, wind=np.full_like(forcing["hs"], 15.0))

    def weathered(r):
        # loss on cells still present at the end, i.e. excluding collapsed blocks
        s = r.snapshots["rock"].astype(np.float32)
        return ((s[:, 0] - s[:, -1]) * (s[:, -1] > 0))[:, 36:].sum()  # rows above +5 m

    _, a = run(init, calm, np.arange(B))
    _, b = run(init, windy, np.arange(B))
    assert weathered(b) > 1.5 * weathered(a)
