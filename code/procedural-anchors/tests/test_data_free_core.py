"""Tests for the data-free latent space core: config, latent, decoder, field."""

import math

import pytest
import torch

from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder, smooth_window
from arm_planner.data_free.field import (
    FieldBatch,
    evaluate_field,
    field_and_grad,
    sample_points,
)
from arm_planner.data_free.latent import (
    chord,
    sample_cluster,
    sample_sphere,
    slerp,
    tangent_basis,
)

CONSTRAINTS = [
    "eikonal", "consistency", "barrier", "origin", "volume", "pose",
    "pose_centroid", "pose_scale", "pose_axes",
    "bridge", "delete", "fill", "widen", "thicken", "grad_change",
]


def small_cfg(**kw):
    base = dict(width=64, layers=8, skip_layer=4, mapping_width=32, points=256)
    base.update(kw)
    return DataFreeConfig(**base)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def box_points(b, n, seed=0):
    return torch.rand(b, n, 3, generator=gen(seed)) * 2 - 1


# ----------------------------------------------------------------- config
def test_config_defaults_and_aliases():
    cfg = DataFreeConfig()
    assert (cfg.R, cfg.R_b, cfg.t, cfg.N) == (cfg.base_radius, cfg.bound_radius, cfg.min_feature, cfg.points)
    assert (cfg.latent_dim, cfg.width, cfg.layers, cfg.skip_layer) == (32, 512, 8, 4)
    # trace of the R = 0.5 ball as the soft indicator (tau = 0.02) measures it
    assert cfg.cov_trace_target == pytest.approx(0.155527, abs=1e-6)
    assert DataFreeConfig(soft_tau=1e-6).cov_trace_target == pytest.approx(3 * 0.5 ** 2 / 5)
    assert cfg.delete_size == pytest.approx(cfg.t / 2)
    assert cfg.delete_gap == pytest.approx(2 * cfg.t)
    assert cfg.r_s() == pytest.approx(2.5 * math.sqrt(4 * math.pi * 0.25 / 2048))
    assert cfg.r_v() == pytest.approx(2.0 * (8 / 2048) ** (1 / 3))
    assert DataFreeConfig(surface_edge_radius=0.2, volume_edge_radius=0.3).r_s() == 0.2
    assert DataFreeConfig(surface_edge_radius=0.2, volume_edge_radius=0.3).r_v() == 0.3
    for name in CONSTRAINTS:
        assert cfg.tolerances[name] > 0
    # default_factory: instances do not share the dict
    assert cfg.tolerances is not DataFreeConfig().tolerances
    with pytest.raises(AttributeError):
        cfg.R = 1.0
    # the bounds that make every shape non-empty and bounded
    assert cfg.use_window is False and cfg.correction_bound == 0.35
    assert cfg.R - cfg.correction_bound > 0 and cfg.R + cfg.correction_bound < cfg.R_b
    with pytest.raises(ValueError):
        DataFreeConfig(correction_bound=0.5)       # not below base_radius
    with pytest.raises(ValueError):
        DataFreeConfig(correction_bound=0.45)      # R + A reaches past R_b
    with pytest.raises(ValueError):
        DataFreeConfig(bound_radius=0.8)
    assert DataFreeConfig.from_dict({**cfg.to_dict(), "unknown": 1}) == cfg


# ----------------------------------------------------------------- latent
def test_sample_sphere_unit():
    z = sample_sphere(100, 32, generator=gen())
    assert z.shape == (100, 32)
    assert torch.allclose(z.norm(dim=-1), torch.ones(100), atol=1e-6)
    assert torch.equal(z, sample_sphere(100, 32, generator=gen()))
    # roughly uniform: mean near zero
    assert sample_sphere(20000, 8, generator=gen()).mean(0).abs().max() < 0.02


def test_chord_formula_and_gradient_at_zero():
    a = sample_sphere(50, 32, generator=gen(1))
    b = sample_sphere(50, 32, generator=gen(2))
    expected = torch.sqrt((2 - 2 * (a * b).sum(-1)).clamp_min(0))
    assert torch.allclose(chord(a, b), expected, atol=1e-6)
    assert torch.allclose(chord(a, b), (a - b).norm(dim=-1), atol=1e-5)
    assert chord(a[:, None], b[None]).shape == (50, 50)  # broadcasting

    a = a.clone().requires_grad_(True)
    same = a.detach().clone().requires_grad_(True)
    d = chord(a, same)
    assert d.abs().max() < 1e-3
    d.sum().backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(same.grad).all()


def test_slerp():
    a = sample_sphere(20, 32, generator=gen(1))
    b = sample_sphere(20, 32, generator=gen(2))
    assert torch.allclose(slerp(a, b, 0.0), a, atol=1e-5)
    assert torch.allclose(slerp(a, b, 1.0), b, atol=1e-5)
    t = torch.rand(20, generator=gen(3))
    mid = slerp(a, b, t)
    assert torch.allclose(mid.norm(dim=-1), torch.ones(20), atol=1e-5)
    # constant angular speed: the angle from a is t times the full angle
    full = torch.acos((a * b).sum(-1))
    assert torch.allclose(torch.acos((a * mid).sum(-1).clamp(-1, 1)), t * full, atol=1e-3)
    # identical codes stay put and stay finite
    assert torch.allclose(slerp(a, a, 0.3), a, atol=1e-5)


def test_sample_cluster():
    center = sample_sphere(1, 32, generator=gen(4))[0]
    for eps in (0.05, 0.5, 1.0):
        codes = sample_cluster(center, 16, eps, generator=gen(5))
        assert codes.shape == (16, 32)
        assert torch.allclose(codes[0], center, atol=1e-6)
        assert torch.allclose(codes.norm(dim=-1), torch.ones(16), atol=1e-5)
        d = chord(codes, center)
        assert d.max() <= eps * (1 + 1e-4)
        # radii are spread, not all at eps
        assert d[1:].min() < 0.25 * eps and d[1:].max() > 0.75 * eps
    # batched centres with one eps each
    centers = sample_sphere(6, 32, generator=gen(6))
    eps = torch.tensor([0.05, 0.1, 0.2, 0.4, 0.8, 1.0])
    codes = sample_cluster(centers, 4, eps, generator=gen(7))
    assert codes.shape == (6, 4, 32)
    assert torch.allclose(codes[:, 0], centers, atol=1e-6)
    assert (chord(codes, centers[:, None]) <= eps[:, None] * (1 + 1e-4)).all()


def test_tangent_basis():
    z = sample_sphere(1, 32, generator=gen(8))[0]
    basis = tangent_basis(z)
    assert basis.shape == (31, 32)
    assert torch.allclose(basis @ basis.T, torch.eye(31), atol=1e-5)
    assert (basis @ z).abs().max() < 1e-5


# ---------------------------------------------------------------- decoder
def test_window_is_c2_and_exact_at_the_ends():
    rho = torch.linspace(0, 1.5, 3001, dtype=torch.float64)
    w = smooth_window(rho, 0.7, 0.9)
    assert (w[rho <= 0.7] == 1).all() and (w[rho >= 0.9] == 0).all()
    assert ((w >= 0) & (w <= 1)).all()
    # first and second derivatives vanish at both ends (checked by autograd)
    for edge in (0.7, 0.9):
        r = torch.tensor([edge - 1e-5, edge + 1e-5], dtype=torch.float64, requires_grad=True)
        (d1,) = torch.autograd.grad(smooth_window(r, 0.7, 0.9).sum(), r, create_graph=True)
        (d2,) = torch.autograd.grad(d1.sum(), r)
        # w'' grows linearly from zero at each end (about 0.075 here, 144 mid-window)
        assert d1.abs().max() < 1e-5 and d2.abs().max() < 0.1


@pytest.mark.parametrize("full_size", [False, True])
def test_init_is_a_sphere(full_size):
    torch.manual_seed(0)
    cfg = DataFreeConfig() if full_size else small_cfg()
    dec = ModulatedSphereDecoder(cfg)
    codes = sample_sphere(4, cfg.latent_dim, generator=gen())
    x = box_points(4, 4096)
    x[:, 0] = 0.0
    with torch.no_grad():
        f = dec(codes, x)
    assert f.shape == (4, 4096)
    deviation = (f - (cfg.R - x.norm(dim=-1))).abs().max()
    # close to the sphere but deliberately not exact (measured at full size on
    # the box: about 5e-3 max, 1e-3 mean), so that codes differ from the start
    assert 1e-4 < deviation < 1e-2
    assert f[:, 0].min() > 0.4  # origin well inside


def randomise(dec, std=0.5):
    """Perturb every parameter, so a test does not only see the near-sphere init."""
    with torch.no_grad():
        for p in dec.parameters():
            p.add_(torch.randn_like(p) * std * (p.std() if p.numel() > 1 and p.std() > 0 else 1.0))
        dec.out.weight.normal_(std=1.0)
        dec.out.bias.normal_(std=1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_code_is_non_empty_and_bounded_for_any_weights(seed):
    torch.manual_seed(seed)
    cfg = small_cfg()
    dec = ModulatedSphereDecoder(cfg)
    randomise(dec)
    assert not dec.use_window
    codes = sample_sphere(64, cfg.latent_dim, generator=gen(seed))
    x = box_points(64, 4096, seed)
    x[:, :1024] *= 0.2  # plenty of points in the inner ball
    x[:, 0] = 0.0
    rho = x.norm(dim=-1)
    a = cfg.correction_bound
    with torch.no_grad():
        f = dec(codes, x)
    c = f - (cfg.R - torch.sqrt(rho ** 2 + 1e-8))
    assert c.abs().max() <= a + 1e-6
    assert c.abs().max() > 0.9 * a and c.min() < -0.2 and c.max() > 0.2  # the bound is exercised
    inner, outer = rho < cfg.R - a, rho > cfg.R + a
    assert inner.sum() > 1000 and outer.sum() > 1000
    assert (f[inner] > 0).all()            # non-empty: the ball of radius R - A is inside
    assert (f[outer] < 0).all()            # bounded: nothing beyond R + A
    assert (f[rho >= cfg.R_b] < 0).all()   # so |x| >= R_b is guaranteed free space
    assert (f[:, 0] >= cfg.R - a - 1e-4).all()


def test_windowed_far_field_is_exact():
    torch.manual_seed(0)
    cfg = small_cfg(use_window=True)
    dec = ModulatedSphereDecoder(cfg)
    assert dec.use_window
    # make the correction large so the test is not vacuous
    with torch.no_grad():
        dec.out.weight.normal_(std=1.0)
    codes = sample_sphere(4, cfg.latent_dim, generator=gen())
    x = box_points(4, 4096)
    rho = torch.sqrt((x * x).sum(-1) + 1e-8)
    far = x.norm(dim=-1) >= cfg.R_b
    assert far.any() and (~far).any()
    with torch.no_grad():
        f = dec(codes, x)
    assert torch.equal(f[far], (cfg.R - rho)[far])
    assert (f[~far] - (cfg.R - rho)[~far]).abs().max() > 1e-2
    assert (f - (cfg.R - rho)).abs().max() <= cfg.correction_bound + 1e-6


def test_no_window_by_default():
    torch.manual_seed(0)
    cfg = small_cfg()
    dec = ModulatedSphereDecoder(cfg)
    with torch.no_grad():
        dec.out.weight.normal_(std=1.0)
    codes = sample_sphere(4, cfg.latent_dim, generator=gen())
    x = box_points(4, 4096)
    far = x.norm(dim=-1) >= cfg.R_b
    with torch.no_grad():
        f = dec(codes, x)
    # the correction is alive in the far field too (no fixed sphere there)
    assert (f[far] - (cfg.R - x.norm(dim=-1))[far]).abs().mean() > 1e-2


def test_far_field_culling_matches_dense_evaluation():
    torch.manual_seed(0)
    cfg = small_cfg(use_window=True)
    dec = ModulatedSphereDecoder(cfg)
    with torch.no_grad():
        dec.out.weight.normal_(std=1.0)
        dec.film.weight.normal_(std=0.05)
    codes = sample_sphere(3, cfg.latent_dim, generator=gen())
    x = box_points(3, 500).requires_grad_(True)
    mod = dec.prepare(codes)
    assert dec.cull_far_field
    f_cull, g_cull = field_and_grad(dec, mod, x, create_graph=False)
    dec.cull_far_field = False
    f_full, g_full = field_and_grad(dec, mod, x, create_graph=False)
    assert torch.allclose(f_cull, f_full, atol=1e-6)
    assert torch.allclose(g_cull, g_full, atol=1e-5)
    dec.cull_far_field = True
    # all points far, and arbitrary leading shape
    assert torch.equal(dec.query(mod, torch.full((3, 5, 3), 0.95)), torch.full((3, 5), cfg.R) - torch.sqrt(torch.tensor(3 * 0.95 ** 2 + 1e-8)))
    assert dec.query(mod, box_points(3, 24).view(3, 4, 6, 3)).shape == (3, 4, 6)


def test_gradient_norm_is_one_at_init():
    torch.manual_seed(0)
    cfg = small_cfg()
    dec = ModulatedSphereDecoder(cfg)
    codes = sample_sphere(4, cfg.latent_dim, generator=gen())
    x = box_points(4, 2048)
    f, g = field_and_grad(dec, dec.prepare(codes), x, create_graph=False)
    away = x.norm(dim=-1) > 0.05
    assert (g.norm(dim=-1)[away] - 1).abs().max() < 2e-2  # measured about 1e-2 at full size
    # positive inside: the gradient points inwards, towards the origin
    cos = torch.nn.functional.cosine_similarity(g, -x, dim=-1)
    assert cos[away].min() > 0.999
    # finite at the origin itself
    _, g0 = field_and_grad(dec, dec.prepare(codes), torch.zeros(4, 1, 3), create_graph=False)
    assert torch.isfinite(g0).all()


def test_codes_are_normalised_inside_the_decoder():
    torch.manual_seed(0)
    cfg = small_cfg()
    dec = ModulatedSphereDecoder(cfg)
    with torch.no_grad():
        dec.out.weight.normal_(std=1.0)
        dec.film.weight.normal_(std=0.05)
    codes = sample_sphere(4, cfg.latent_dim, generator=gen())
    x = box_points(4, 512) * 0.5
    with torch.no_grad():
        f1 = dec(codes, x)
        f2 = dec(codes * 7.5, x)
        f3 = dec(codes.roll(1, 0), x)
    assert torch.allclose(f1, f2, atol=1e-5)
    assert (f1 - f3).abs().max() > 1e-3  # and the code does matter


def test_double_backward_reaches_parameters():
    torch.manual_seed(0)
    cfg = small_cfg()
    dec = ModulatedSphereDecoder(cfg)
    codes = sample_sphere(4, cfg.latent_dim, generator=gen())
    x = box_points(4, 256) * 0.6
    f, g = field_and_grad(dec, dec.prepare(codes), x, create_graph=True)
    assert g.requires_grad
    loss = ((g.norm(dim=-1) - 1) ** 2).mean()
    loss.backward()
    for name, p in dec.named_parameters():
        if name == "out.bias":  # a constant offset has no spatial gradient
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for name, p in dec.named_parameters():
        if name != "out.bias":
            assert p.grad.abs().max() > 0, name


def test_film_init_differs_between_codes():
    torch.manual_seed(0)
    cfg = small_cfg()
    dec = ModulatedSphereDecoder(cfg)
    with torch.no_grad():
        mod = dec.prepare(sample_sphere(2000, cfg.latent_dim, generator=gen()))
    for part in (mod.scale, mod.shift):
        assert part.shape == (2000, cfg.layers, cfg.width)
        assert part.std(dim=0).mean().item() == pytest.approx(cfg.film_init_std, rel=0.1)
        assert part.mean(dim=0).abs().max() < 0.03
    # film_init_std = 0 gives the old identical-at-init behaviour
    dec0 = ModulatedSphereDecoder(small_cfg(film_init_std=0.0))
    with torch.no_grad():
        mod0 = dec0.prepare(sample_sphere(4, cfg.latent_dim, generator=gen()))
    assert (mod0.scale == 0).all() and (mod0.shift == 0).all()


@pytest.mark.parametrize("full_size", [False, True])
def test_diversity_gradient_is_alive_at_init(full_size):
    torch.manual_seed(0)
    cfg = DataFreeConfig() if full_size else small_cfg()
    dec = ModulatedSphereDecoder(cfg)
    center = sample_sphere(1, cfg.latent_dim, generator=gen())[0]
    codes = sample_cluster(center, 2, 0.3, generator=gen(1))  # two codes of one cluster
    assert 0 < chord(codes[0], codes[1]) <= 0.3
    x = box_points(1, 1024).expand(2, -1, -1)  # shared points
    f = dec(codes, x)
    diff = (f[0] - f[1]).abs().mean()
    assert diff > 1e-6  # far above float32 noise on values of order 1 (about 1e-7)
    diff.backward()
    for name, p in dec.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for name in ("out.weight", "hidden.0.weight", "hidden.7.weight", "film.weight", "mapping.0.weight"):
        assert dict(dec.named_parameters())[name].grad.abs().max() > 0, name


def test_field_and_grad_matches_finite_differences_and_no_grad():
    torch.manual_seed(0)
    cfg = small_cfg()
    dec = ModulatedSphereDecoder(cfg).double()
    with torch.no_grad():
        dec.out.weight.normal_(std=0.5)
    codes = sample_sphere(2, cfg.latent_dim, generator=gen()).double()
    x = box_points(2, 64).double() * 0.6
    with torch.no_grad():  # must work when the caller has gradients off
        mod = dec.prepare(codes)
        f, g = field_and_grad(dec, mod, x, create_graph=False)
        h = 1e-5
        for axis in range(3):
            step = torch.zeros(3, dtype=torch.float64)
            step[axis] = h
            fd = (dec.query(mod, x + step) - dec.query(mod, x - step)) / (2 * h)
            assert torch.allclose(g[..., axis], fd, atol=1e-6)
    assert not g.requires_grad


# ------------------------------------------------------------------ field
def test_sample_points():
    cfg = small_cfg()
    x = sample_points(cfg, 5, gen(), torch.device("cpu"))
    assert x.shape == (5, cfg.N, 3)
    assert (x[:, 0] == 0).all()
    assert x.min() >= -1 and x.max() <= 1
    assert x[:, 1:].abs().max() > 0.9 and x.mean().abs() < 0.05
    assert torch.equal(x, sample_points(cfg, 5, gen(), torch.device("cpu")))

    extra = torch.full((5, 7, 3), float("nan"))
    extra[0, :3] = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [1.5, -2.0, 0.0]])
    extra[2, 5] = torch.tensor([-0.7, 0.7, 0.0])
    y = sample_points(cfg, 5, gen(), torch.device("cpu"), extra=extra)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert (y[:, 0] == 0).all() and y.min() >= -1 and y.max() <= 1
    tail = y[:, -7:]
    assert torch.equal(tail[0, :2], extra[0, :2])
    assert torch.equal(tail[0, 2], torch.tensor([1.0, -1.0, 0.0]))  # clamped to the box
    assert torch.equal(tail[2, 5], extra[2, 5])
    # everything else is untouched
    valid = torch.isfinite(extra).all(-1)
    assert torch.equal(tail[~valid], x[:, -7:][~valid])
    assert torch.equal(y[:, :-7], x[:, :-7])
    # more extra points than rows: the origin survives
    z = sample_points(small_cfg(points=4), 2, gen(), torch.device("cpu"), extra=torch.zeros(2, 9, 3) + 0.5)
    assert (z[:, 0] == 0).all() and (z[:, 1:] == 0.5).all()


@pytest.mark.parametrize("detach", [False, True])
def test_evaluate_field(detach):
    torch.manual_seed(0)
    cfg = small_cfg(clusters=3, codes_per_cluster=4, consistency_detach_direction=detach)
    dec = ModulatedSphereDecoder(cfg)
    with torch.no_grad():
        dec.out.weight.normal_(std=0.003)
        dec.film.weight.normal_(std=0.02)
    g_ = gen()
    centers = sample_sphere(cfg.clusters, cfg.latent_dim, generator=g_)
    codes = sample_cluster(centers, cfg.codes_per_cluster, 0.3, generator=g_).reshape(-1, cfg.latent_dim)
    cluster = torch.arange(cfg.clusters).repeat_interleave(cfg.codes_per_cluster)
    x = sample_points(cfg, cfg.clusters, g_, torch.device("cpu"))
    b, n = cfg.batch_size, cfg.N

    fb = evaluate_field(dec, codes * 3.0, cluster, x, cfg)
    assert isinstance(fb, FieldBatch)
    assert fb.codes.shape == (b, cfg.latent_dim) and fb.cluster.shape == (b,)
    assert fb.cluster.dtype == torch.long
    assert torch.allclose(fb.codes.norm(dim=-1), torch.ones(b), atol=1e-6)
    assert fb.x.shape == (b, n, 3) and fb.f.shape == (b, n) and fb.g.shape == (b, n, 3)
    assert fb.xp.shape == (b, n, 3) and fb.fp.shape == (b, n)
    # shared points within a cluster, origin first
    assert torch.equal(fb.x, x[cluster])
    assert (fb.x[:, 0] == 0).all()
    assert not fb.x.requires_grad
    # xp == x - f g
    assert torch.allclose(fb.xp, fb.x - fb.f[..., None] * fb.g, atol=1e-6)
    assert torch.allclose(fb.fp, dec.query(fb.mod, fb.xp), atol=1e-6)
    assert fb.f.requires_grad and fb.g.requires_grad and fb.fp.requires_grad
    # the guesses land much closer to the surface than the samples
    assert fb.fp.abs().mean() < 0.2 * fb.f.abs().mean()

    # Eikonal + consistency back-propagate to the parameters
    loss = ((fb.g.norm(dim=-1) - 1) ** 2).mean() + fb.fp.abs().mean()
    loss.backward()
    grads = [p.grad for p in dec.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(q).all() for q in grads)
    assert dec.hidden[0].weight.grad.abs().max() > 0

    # per-code points are accepted too
    fb2 = evaluate_field(dec, codes, cluster, x[cluster], cfg, create_graph=False)
    assert torch.allclose(fb2.f, fb.f, atol=1e-5)
    assert not fb2.g.requires_grad and fb2.f.requires_grad
    with pytest.raises(ValueError):
        evaluate_field(dec, codes, cluster, x[:2], cfg)


def test_detach_direction_keeps_the_step_length_in_the_graph():
    torch.manual_seed(0)
    cfg = small_cfg(clusters=2, codes_per_cluster=2, consistency_detach_direction=True)
    dec = ModulatedSphereDecoder(cfg)
    codes = sample_sphere(4, cfg.latent_dim, generator=gen())
    cluster = torch.tensor([0, 0, 1, 1])
    fb = evaluate_field(dec, codes, cluster, sample_points(cfg, 2, gen(), torch.device("cpu")), cfg)
    assert fb.xp.requires_grad  # through f


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_smoke():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    cfg = small_cfg(clusters=2, codes_per_cluster=2)
    dec = ModulatedSphereDecoder(cfg).to(dev)
    cpu_gen = gen()
    centers = sample_sphere(2, cfg.latent_dim, generator=cpu_gen, device=dev)
    codes = sample_cluster(centers, 2, 0.2, generator=cpu_gen).reshape(-1, cfg.latent_dim)
    x = sample_points(cfg, 2, cpu_gen, dev)
    assert codes.device.type == "cuda" and x.device.type == "cuda"
    cuda_gen = torch.Generator(device=dev).manual_seed(0)
    assert sample_points(cfg, 2, cuda_gen, dev).device.type == "cuda"
    fb = evaluate_field(dec, codes, torch.tensor([0, 0, 1, 1]), x, cfg)
    (((fb.g.norm(dim=-1) - 1) ** 2).mean() + fb.fp.abs().mean()).backward()
    assert torch.isfinite(dec.out.weight.grad).all()


def test_activation_leak_keeps_a_slope_when_a_unit_is_off_and_is_off_by_default():
    torch.manual_seed(4)
    plain = ModulatedSphereDecoder(small_cfg())
    leaky = ModulatedSphereDecoder(small_cfg(activation_leak=0.05))
    leaky.load_state_dict(plain.state_dict())
    value = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0], requires_grad=True)
    soft = torch.nn.functional.softplus(value, beta=plain.beta)
    assert torch.equal(plain.activation(value), soft)                       # default: plain softplus
    out = leaky.activation(value)
    assert torch.allclose(out, 0.05 * value + 0.95 * soft)
    (slope,) = torch.autograd.grad(out.sum(), value, create_graph=True)
    assert float(slope[0]) == pytest.approx(0.05, abs=1e-6)                  # far negative: slope is the leak, not zero
    assert float(slope[-1]) == pytest.approx(1.0, abs=1e-6)
    (curve,) = torch.autograd.grad(slope.sum(), value)                       # still smooth: a second derivative exists
    assert float(curve[2]) > 0
    (dead,) = torch.autograd.grad(plain.activation(value).sum(), value)
    assert float(dead[0]) < 1e-20                                            # the plain activation is flat there
    codes = sample_sphere(3, plain.latent_dim, generator=gen())
    x = torch.rand(3, 50, 3, generator=gen()) * 2 - 1
    assert not torch.allclose(plain(codes, x), leaky(codes, x))


def test_fourier_bands_widen_the_coordinate_input_and_keep_the_sphere_start_and_second_derivatives():
    torch.manual_seed(6)
    plain = ModulatedSphereDecoder(small_cfg())
    dec = ModulatedSphereDecoder(small_cfg(fourier_bands=3))
    assert plain.hidden[0].in_features == 3 and plain.hidden[4].in_features == plain.width + 3
    assert dec.hidden[0].in_features == 21 and dec.hidden[4].in_features == dec.width + 21
    x = (torch.rand(2, 30, 3, generator=gen()) * 2 - 1).requires_grad_(True)
    enc = dec.encode(x)
    assert enc.shape == (2, 30, 21) and torch.equal(enc[..., :3], x)
    assert torch.allclose(enc[..., 3:12], torch.sin(x[..., None] * (math.pi * torch.tensor([1.0, 2.0, 4.0]))).flatten(-2))
    assert torch.allclose(enc[..., 12:], torch.cos(x[..., None] * (math.pi * torch.tensor([1.0, 2.0, 4.0]))).flatten(-2))
    assert torch.equal(plain.encode(x), x)                              # off by default: raw coordinates
    codes = sample_sphere(2, dec.latent_dim, generator=gen())
    mod = dec.prepare(codes)
    f = dec.query(mod, x)
    rho = x.norm(dim=-1)
    assert float((f - (dec.base_radius - rho)).abs().max()) < 0.05      # still starts close to the sphere
    f, g = field_and_grad(dec, mod, x, create_graph=True)
    grads = torch.autograd.grad((g.norm(dim=-1) - 1.0).square().mean(), list(dec.parameters()), allow_unused=True)
    assert all(torch.isfinite(v).all() for v in grads if v is not None)
    assert "fourier_freq" not in dec.state_dict()                       # old checkpoints stay loadable


def test_layer_norm_centres_the_chosen_layers_and_keeps_second_derivatives():
    torch.manual_seed(5)
    plain = ModulatedSphereDecoder(small_cfg())
    last = ModulatedSphereDecoder(small_cfg(layer_norm_layers=1))
    every = ModulatedSphereDecoder(small_cfg(layer_norm_layers=8))
    last.load_state_dict(plain.state_dict())
    every.load_state_dict(plain.state_dict())
    assert (plain.norm_from, last.norm_from, every.norm_from) == (8, 7, 0)
    codes = sample_sphere(3, plain.latent_dim, generator=gen())
    x = (torch.rand(3, 40, 3, generator=gen()) * 2 - 1).requires_grad_(True)
    assert not torch.allclose(plain(codes, x), last(codes, x))
    for dec, normed in ((plain, []), (last, [7]), (every, list(range(8)))):
        mod = dec.prepare(codes)
        seen = []
        raw = dec._net(mod, x, collect=seen)
        assert len(seen) == 8 and seen[0].shape == (3, 40, dec.width)
        assert torch.equal(raw, dec._net(mod, x))                      # collecting changes nothing
        for layer in range(8):
            g = 1.0 + mod.scale[:, layer][:, None, :]
            undone = (seen[layer] - mod.shift[:, layer][:, None, :]) / g      # the value before the code's scale and shift
            centred = float(undone.mean(-1).abs().max()) < 1e-4 and float((undone.var(-1, unbiased=False) - 1).abs().max()) < 1e-2
            assert centred == (layer in normed)
    f, g = field_and_grad(every, every.prepare(codes), x, create_graph=True)
    loss = (g.norm(dim=-1) - 1.0).square().mean()                       # an eikonal-style term needs a second derivative
    grads = torch.autograd.grad(loss, list(every.parameters()), allow_unused=True)
    assert all(torch.isfinite(v).all() for v in grads if v is not None)
    assert sum(float(v.abs().sum()) for v in grads if v is not None) > 0


def test_inner_bound_frees_the_centre_and_keeps_the_outer_limit():
    """With correction_bound_inner the origin may lie outside the shape; radius R + A still bounds it."""
    torch.manual_seed(3)
    cfg = small_cfg(correction_bound_inner=1.35)
    dec = ModulatedSphereDecoder(cfg)
    plain = ModulatedSphereDecoder(small_cfg())
    plain.load_state_dict(dec.state_dict())
    codes = sample_sphere(4, cfg.latent_dim, generator=gen())
    x = torch.rand(4, 400, 3, generator=gen()) * 2 - 1
    x[:, 0] = 0.0
    # near the sphere (small raw output) the two forms agree closely
    assert torch.allclose(dec(codes, x), plain(codes, x), atol=1e-4)
    with torch.no_grad():
        dec.out.weight.mul_(1e4)                      # huge raw outputs, of both signs
        dec.out.bias.fill_(-50.0)                     # ... and strongly negative at most points
        plain.load_state_dict(dec.state_dict())
    f, old = dec(codes, x), plain(codes, x)
    rho = x.norm(dim=-1)
    outer = cfg.base_radius + cfg.correction_bound
    assert (f <= outer - rho + 1e-3).all()            # never inside beyond radius R + A
    assert (f >= cfg.base_radius - 1.35 - rho - 1e-3).all()
    assert (old[:, 0] > 0).all()                      # symmetric form: the centre is always filled
    assert (f[:, 0] < -0.5).any()                     # freed form: the centre can be well outside
    # the join at zero is smooth: gradients stay finite and the field can be trained
    xg = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(dec(codes, xg).sum(), xg)
    assert torch.isfinite(grad).all()
    with pytest.raises(ValueError):
        small_cfg(correction_bound_inner=0.0)


def test_fixed_surface_consistency_blocks_only_the_landing_route():
    """Full gradient of the landing value = route through the starting point + route through the landing point."""
    torch.manual_seed(5)
    cfg = small_cfg()
    fixed_cfg = small_cfg(consistency_fixed_surface=True, consistency_min_depth=0.05)
    dec = ModulatedSphereDecoder(cfg).double()
    with torch.no_grad():
        dec.out.weight.mul_(200.0)                    # far enough from the sphere for the routes to differ
    codes = sample_sphere(3, cfg.latent_dim, generator=gen()).double()
    x = (torch.rand(1, 60, 3, generator=gen()) * 2 - 1).double()
    x[:, 0] = 0.0
    cluster = torch.zeros(3, dtype=torch.long)
    params = [p for p in dec.parameters()]

    def grads(value):
        got = torch.autograd.grad(value, params, allow_unused=True, retain_graph=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(got, params)])

    full = evaluate_field(dec, codes, cluster, x, cfg)
    fixed = evaluate_field(dec, codes, cluster, x, fixed_cfg)
    assert torch.allclose(full.fp, fixed.fp)          # same values, different gradient routes
    landing = dec.query(dec.prepare(codes), full.xp.detach())
    g_full, g_fixed, g_landing = grads(full.fp.sum()), grads(fixed.fp.sum()), grads(landing.sum())
    assert torch.allclose(g_full, g_fixed + g_landing, rtol=1e-6, atol=1e-9)
    assert float(g_landing.norm()) > 1e-3 and float(g_fixed.norm()) > 1e-3
    # starting points close to the surface are left out of the term
    assert fixed.fp_weight is not None and full.fp_weight is None
    near = fixed.f.detach().abs() < 0.05
    assert bool(near.any()) and bool((fixed.fp_weight[near] == 0).all()) and bool((fixed.fp_weight[~near] == 1).all())
    from arm_planner.data_free import losses as L
    w = fixed.fp_weight[:, 1:]
    assert float(L.consistency(fixed)) == pytest.approx(float((fixed.fp[:, 1:].abs() * w).sum() / w.sum()))

    # field_moves_surface: a push on the value at x also acts on the field where x's step lands
    moved = evaluate_field(dec, codes, cluster, x, small_cfg(field_moves_surface=True))
    assert moved.f_move is not None and torch.allclose(moved.f_move, full.f) and torch.allclose(moved.fp, full.fp)
    g_value = grads(full.f.sum())
    assert torch.allclose(grads(moved.f_move.sum()), g_value + g_landing, rtol=1e-6, atol=1e-9)
    assert torch.allclose(grads(moved.fp.sum()), g_full, rtol=1e-6, atol=1e-9)      # both routes, as before
    both = evaluate_field(dec, codes, cluster, x, small_cfg(field_moves_surface=True, consistency_fixed_surface=True,
                                                             consistency_min_depth=0.05))
    assert torch.allclose(both.fp, full.fp)
    assert torch.allclose(grads(both.fp.sum()), g_fixed, rtol=1e-6, atol=1e-9)      # only the starting-point route
    assert torch.allclose(grads(both.f_move.sum()), g_value + g_landing, rtol=1e-6, atol=1e-9)
    assert torch.equal(both.fp_weight, fixed.fp_weight)

    # the dial: the landing route at 30% strength, the starting-point route at full strength
    dial = evaluate_field(dec, codes, cluster, x, small_cfg(consistency_surface_scale=0.3))
    assert torch.allclose(dial.fp, full.fp) and dial.fp_weight is None and dial.f_move is None
    assert torch.allclose(grads(dial.fp.sum()), g_fixed + 0.3 * g_landing, rtol=1e-6, atol=1e-9)
    off = evaluate_field(dec, codes, cluster, x, small_cfg(consistency_surface_scale=0.0))
    assert torch.allclose(grads(off.fp.sum()), g_fixed, rtol=1e-6, atol=1e-9)
