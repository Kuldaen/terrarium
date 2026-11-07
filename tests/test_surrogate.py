import numpy as np
import pytest

torch = pytest.importorskip("torch")

from terrarium.phenomena.erosion.cliff2d.surrogate import (  # noqa: E402
    N_COND,
    FlowSurrogate,
    SurrogateConfig,
    cond_images,
    decode_target,
    encode_target,
)


@pytest.mark.parametrize("use_strength", [True, False])
def test_loss_and_sampling_shapes(use_strength):
    cfg = SurrogateConfig(base_ch=8, use_strength=use_strength)
    m = FlowSurrogate(cfg, np.zeros(N_COND), np.ones(N_COND), torch.device("cpu"))
    rock0 = np.ones((2, 80, 160), np.float32)
    cimg = torch.tensor(cond_images(rock0, rock0 if use_strength else None))
    cvec = torch.zeros(2, N_COND)
    x1 = torch.tensor(encode_target(rock0, rock0 * 0.5, np.array([1.0, 2.0])))
    assert torch.isfinite(m.loss(x1, cimg, cvec))
    x = m.sample(cimg, cvec, steps=2)
    assert x.shape == (2, 2, 80, 160)


def test_decode_never_adds_rock():
    rng = np.random.default_rng(0)
    rock0 = (rng.random((3, 80, 160)) > 0.5).astype(np.float32)
    x = rng.normal(size=(3, 2, 80, 160)).astype(np.float32) * 2
    rock1, talus = decode_target(x, rock0)
    assert (rock1 <= rock0 + 1e-6).all() and (rock1 >= 0).all() and (talus >= 0).all()


def test_target_roundtrip():
    rock0 = np.ones((1, 80, 160), np.float32)
    rock1 = rock0.copy()
    rock1[0, 40:, :20] = 0.0
    r, t = decode_target(encode_target(rock0, rock1, np.array([5.0])), rock0)
    np.testing.assert_allclose(r, rock1, atol=1e-6)
    np.testing.assert_allclose(t, [5.0], atol=1e-4)


def _tiny_model():
    from terrarium.phenomena.erosion.cliff2d.surrogate import FlowSurrogate, SurrogateConfig

    return FlowSurrogate(SurrogateConfig(base_ch=8), np.zeros(N_COND), np.ones(N_COND), torch.device("cpu"))


def _start(n=2):
    from terrarium.phenomena.erosion.cliff2d import generate_cliffs

    c = generate_cliffs(n, np.random.default_rng(0))
    return dict(rock0=c["rock"], strength=c["strength"], top_row=c["top_row"], talus0=np.zeros(n, np.float32))


def test_rollout_shapes_and_monotone():
    from terrarium.phenomena.erosion.cliff2d.summary import SUMMARY_FEATURES
    from terrarium.phenomena.erosion.cliff2d.surrogate import rollout

    st = _start()
    S = 3
    r = rollout(_tiny_model(), **st, step_days=[360] * S, summaries=np.zeros((2, S, len(SUMMARY_FEATURES)), np.float32),
                rev_plan=np.full((2, S), np.nan, np.float32), n_samples=2, flow_steps=2)
    assert r["rock"].shape == (2, 2, S + 1, 80, 160)
    assert r["outcomes"].shape == (2, 2, S, 3)
    rk = r["rock"].astype(np.float32)
    assert (np.diff(rk, axis=2) <= 1e-3).all()  # rock never comes back along a trajectory


def test_rollout_common_random_numbers():
    from terrarium.phenomena.erosion.cliff2d.summary import SUMMARY_FEATURES
    from terrarium.phenomena.erosion.cliff2d.surrogate import rollout

    st, m = _start(), _tiny_model()
    kw = dict(step_days=[360, 360], summaries=np.zeros((2, 2, len(SUMMARY_FEATURES)), np.float32), n_samples=2, flow_steps=2, keep_fields=False)
    plan = np.full((2, 2), np.nan, np.float32)
    a = rollout(m, **st, rev_plan=plan, seed=7, **kw)["outcomes"]
    b = rollout(m, **st, rev_plan=plan, seed=7, **kw)["outcomes"]
    np.testing.assert_array_equal(a, b)  # same seed + same plan = identical futures


def test_simulate_plan_switches_action(tmp_path):
    from terrarium.phenomena.erosion.cliff2d import generate_cliffs, generate_forcing
    from terrarium.phenomena.erosion.cliff2d.dataset import simulate_plan

    rng = np.random.default_rng(0)
    init = generate_cliffs(4, rng)
    f = generate_forcing(4, 720, rng)
    none = np.full((4, 2), np.nan, np.float32)
    armour = np.full((4, 2), 3.0, np.float32)
    seeds = np.arange(4)
    r0, _ = simulate_plan(init, init["rock"], np.zeros(4), f, seeds, [360, 360], none, device="cpu")
    r1, _ = simulate_plan(init, init["rock"], np.zeros(4), f, seeds, [360, 360], armour, device="cpu")
    lost = lambda r: (r[:, 0].astype(np.float32) - r[:, -1].astype(np.float32))[:, :28].sum()
    assert lost(r1) < lost(r0)
