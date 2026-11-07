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
