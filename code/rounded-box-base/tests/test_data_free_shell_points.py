"""point_sampling = "shells": band counts, radii, origin row, and the uniform default left unchanged."""
import math
import pytest
import torch

from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.field import sample_points


def _cfg(**kw):
    return DataFreeConfig(**kw)


def test_uniform_default_is_unchanged():
    cfg = _cfg()
    assert cfg.point_sampling == "uniform"
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    x = sample_points(cfg, 2, g1, "cpu")
    ref = torch.rand((2, cfg.points, 3), generator=g2) * 2.0 - 1.0
    ref[:, 0] = 0.0
    assert torch.equal(x, ref)


def test_shell_bands_and_radii():
    cfg = _cfg(point_sampling="shells", points=2048)
    x = sample_points(cfg, 3, torch.Generator().manual_seed(1), "cpu")
    assert x.shape == (3, 2048, 3)
    assert torch.all(x[:, 0] == 0.0)
    r = x[:, 1:].norm(dim=-1)
    assert float(r.max()) <= cfg.bound_radius + 1e-5
    edges = (0.0, 0.2, 0.55, 0.7, cfg.bound_radius + 1e-5)
    expect = (round(0.05 * 2048), round(0.5 * 2048), round(0.2 * 2048))
    expect = (*expect, 2048 - sum(expect))
    for c in range(3):
        counts = [int(((r[c] >= lo) & (r[c] < hi)).sum()) for lo, hi in zip(edges, edges[1:])]
        # row 0 (the origin) replaced one point of some band
        assert sum(counts) == 2047
        assert all(abs(a - b) <= 1 for a, b in zip(counts, expect))
    # bands are mixed along the rows (the shared prefix of rows is not one band)
    first = x[0, 1:1025].norm(dim=-1)
    assert float((first < 0.55).float().mean()) == pytest.approx(0.55, abs=0.06)


def test_shell_points_with_hard_negatives():
    cfg = _cfg(point_sampling="shells", points=64)
    extra = torch.full((1, 4, 3), float("nan"))
    extra[0, :2] = torch.tensor([[0.95, 0.0, 0.0], [0.0, -0.3, 0.1]])
    x = sample_points(cfg, 1, torch.Generator().manual_seed(0), "cpu", extra=extra)
    assert torch.allclose(x[0, 60:62], extra[0, :2])


def test_bad_point_sampling_rejected():
    with pytest.raises(ValueError):
        _cfg(point_sampling="grid")


def _ball_batch(x, centre, radius, weight=None):
    from types import SimpleNamespace
    f = radius - (x - torch.tensor(centre)).norm(dim=-1)
    return SimpleNamespace(f=f, x=x, volume_weight=weight)


def test_volume_weight_turns_shell_means_into_box_means():
    from arm_planner.data_free.field import shell_volume_weight
    cfg = _cfg(point_sampling="shells", points=4096)
    x = torch.cat([sample_points(cfg, 1, torch.Generator().manual_seed(s), "cpu")[0, 1:] for s in range(40)])[None]
    w = shell_volume_weight(x, cfg)
    ball = (4.0 / 3.0) * math.pi * cfg.bound_radius ** 3 / 8.0          # box mean of 1 inside R_b
    assert float(w.mean()) == pytest.approx(ball, rel=0.02)


def test_pose_and_volume_estimates_match_uniform_with_volume_weights():
    from arm_planner.data_free import losses as L
    from arm_planner.data_free.field import shell_volume_weight
    n = 4096
    shells = _cfg(point_sampling="shells", points=n)
    uniform = _cfg(points=n)
    xs = torch.cat([sample_points(shells, 1, torch.Generator().manual_seed(s), "cpu")[0, 1:] for s in range(60)])[None]
    xu = torch.cat([sample_points(uniform, 1, torch.Generator().manual_seed(s), "cpu")[0, 1:] for s in range(60)])[None]
    centre, radius = (0.25, -0.1, 0.05), 0.3
    fu = _ball_batch(xu, centre, radius)
    fw = _ball_batch(xs, centre, radius, shell_volume_weight(xs, shells))
    fn = _ball_batch(xs, centre, radius)                                  # shells, unweighted: biased
    (au, cu, ku), (aw, cw, kw), (an, cn, kn) = (L.moments(b, uniform) for b in (fu, fw, fn))
    tr = lambda k: float(torch.diagonal(k[0]).sum())
    assert float(aw[0]) == pytest.approx(float(au[0]), rel=0.05)
    assert torch.allclose(cw[0], cu[0], atol=0.01)
    assert tr(kw) == pytest.approx(tr(ku), rel=0.05)
    # without the weights the shells misread the same ball
    assert abs(float(an[0]) - float(au[0])) > 0.2 * float(au[0])
    assert abs(float(cn[0, 0]) - float(cu[0, 0])) > 0.015


def test_evaluate_field_attaches_volume_weight_only_for_shells():
    from arm_planner.data_free.decoder import ModulatedSphereDecoder
    from arm_planner.data_free.field import evaluate_field
    from arm_planner.data_free.latent import sample_sphere
    for mode in ("uniform", "shells"):
        cfg = _cfg(point_sampling=mode, points=256)
        torch.manual_seed(0)
        dec = ModulatedSphereDecoder(cfg)
        codes = sample_sphere(3, cfg.latent_dim, generator=torch.Generator().manual_seed(1))
        x = sample_points(cfg, 1, torch.Generator().manual_seed(2), "cpu")
        fb = evaluate_field(dec, codes, torch.zeros(3, dtype=torch.long), x, cfg, create_graph=False)
        if mode == "uniform":
            assert fb.volume_weight is None
        else:
            assert fb.volume_weight.shape == fb.f.shape
            assert float(fb.volume_weight.min()) > 0.0
