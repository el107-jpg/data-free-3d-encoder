"""Tests for arm_planner.data_free.losses and .multipliers.

Most tests build a FieldBatch by hand from an analytic field (gradients from
autograd), so they do not depend on the decoder.  The last group goes through
the real decoder and evaluate_field.
"""

import dataclasses
import math

import pytest
import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free.field import FieldBatch, evaluate_field, sample_points
from arm_planner.data_free.latent import chord, sample_cluster, sample_sphere
from arm_planner.data_free.multipliers import AdaptiveMultipliers

R = 0.5


# ------------------------------------------------------------------ helpers
def box_points(batch, count, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, count, 3, generator=gen) * 2.0 - 1.0
    x[:, 0] = 0.0
    return x


def unit_codes(batch, dim=32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, dim, generator=gen)
    return z / z.norm(dim=-1, keepdim=True)


def make_fb(field_fn, x, codes=None, cluster=None):
    """FieldBatch from an analytic field ``field_fn(x: B×N×3) -> B×N``."""
    batch = x.shape[0]
    x = x.clone().requires_grad_(True)
    f = field_fn(x)
    (g,) = torch.autograd.grad(f.sum(), x, create_graph=True)
    xp = x - f.unsqueeze(-1) * g
    fp = field_fn(xp)
    if codes is None:
        codes = unit_codes(batch)
    if cluster is None:
        cluster = torch.arange(batch)
    return FieldBatch(codes=codes, cluster=cluster, x=x, f=f, g=g, xp=xp, fp=fp)


def fb_from_values(f, x, codes, cluster):
    """FieldBatch whose ``f`` is given directly (for losses that only read f)."""
    g = torch.zeros_like(x)
    return FieldBatch(codes=codes, cluster=cluster, x=x, f=f, g=g, xp=x.clone(), fp=f.detach().clone())


def sphere(radius=R, center=(0.0, 0.0, 0.0), gain=1.0):
    c = torch.tensor(center)

    def field(x):
        return gain * (radius - torch.sqrt((x - c).square().sum(-1) + 1e-16))

    return field


def ellipsoid(a, b, c):
    """First-order distance to an axis-aligned ellipsoid, positive inside."""
    axes = torch.tensor([a, b, c])

    def field(x):
        k0 = torch.sqrt((x / axes).square().sum(-1) + 1e-12)
        k1 = torch.sqrt((x / axes.square()).square().sum(-1) + 1e-12)
        return k0 * (1.0 - k0) / k1

    return field


def two_spheres(radius=0.3, offset=0.45):
    a = sphere(radius, (offset, 0.0, 0.0))
    b = sphere(radius, (-offset, 0.0, 0.0))
    return lambda x: torch.maximum(a(x), b(x))


@pytest.fixture(scope="module")
def cfg():
    return DataFreeConfig()


# ------------------------------------------------------- exact sphere is free
POSE_TOLERANCE_4096 = 5e-3


def test_sphere_sdf_satisfies_every_constraint(cfg):
    assert cfg.base_radius == R
    assert cfg.cov_trace_target == pytest.approx(L.soft_ball_trace(R, cfg.soft_tau))
    worst_pose = 0.0
    for seed in range(8):
        fb = make_fb(sphere(), box_points(4, 4096, seed=seed))
        assert float(L.eikonal(fb)) < 1e-10
        assert float(L.consistency(fb)) < 1e-6
        assert float(L.consistency_error(fb)) < 1e-6
        assert float(L.barrier(fb, cfg)) == 0.0
        assert float(L.origin_anchor(fb, cfg)) == 0.0
        assert float(L.min_volume(fb, cfg)) == 0.0
        worst_pose = max(worst_pose, float(L.pose(fb, cfg)))
    print(f"worst pose on exact sphere, N=4096, 8 seeds x 4 codes: {worst_pose:.3e}")
    assert worst_pose < POSE_TOLERANCE_4096


def test_pose_floor_with_many_points(cfg):
    # With the sampling noise gone nothing is left: the default trace target is
    # the trace of the ball as the soft inside indicator measures it (3.7% above
    # 3 R^2 / 5 at tau = 0.02, R = 0.5; against the hard-ball target the pose
    # value would be near 3e-5).
    fb = make_fb(sphere(), box_points(1, 400_000, seed=3))
    value = float(L.pose(fb, cfg))
    print(f"pose on exact sphere, N=400000: {value:.3e}")
    assert value < 5e-6
    assert value < cfg.tolerances["pose"]


def test_soft_ball_trace_matches_quadrature():
    import numpy as np

    for radius, tau in ((0.5, 0.02), (0.5, 0.05), (0.3, 0.01)):
        r = np.linspace(0.0, 3.0, 600_001)
        s = 1.0 / (1.0 + np.exp(-(radius - r) / tau))
        exact = (r ** 4 * s).sum() / (r ** 2 * s).sum()
        assert L.soft_ball_trace(radius, tau) == pytest.approx(exact, rel=1e-5)
    assert L.soft_ball_trace(0.5, 1e-9) == pytest.approx(0.15)


# ------------------------------------------------ unbiased split-sample pose
def _pose_term_means(field_fn, cfg, seeds=60, count=2048, batch=16):
    sums = {key: [] for key in L.POSE_KEYS}
    for seed in range(seeds):
        terms = L.pose_terms(make_fb(field_fn, box_points(batch, count, seed=seed)), cfg)
        for key in L.POSE_KEYS:
            sums[key].append(float(terms[key]))
    values = {key: torch.tensor(v, dtype=torch.float64) for key, v in sums.items()}
    return ({key: float(v.mean()) for key, v in values.items()},
            {key: float(v.std() / len(v) ** 0.5) for key, v in values.items()})


def test_pose_terms_are_unbiased_on_the_exact_sphere(cfg):
    assert set(L.pose_terms(make_fb(sphere(), box_points(2, 256)), cfg)) == set(L.POSE_KEYS)
    mean, error = _pose_term_means(sphere(), cfg)
    print("pose terms on the exact sphere, N=2048:", mean, "standard errors:", error)
    for key in L.POSE_KEYS:
        # zero within four standard errors, and far below the tolerance: the
        # biased single-sample pose reads 1e-3 here
        assert abs(mean[key]) < 4 * error[key] + 2e-6, key
        assert abs(mean[key]) < 0.5 * cfg.tolerances[key], key
    biased = float(L.pose(make_fb(sphere(), box_points(16, 2048)), cfg))
    assert biased > 3 * max(cfg.tolerances[k] for k in L.POSE_KEYS)


def test_soft_ball_moment_matches_the_trace_and_the_hard_ball():
    assert L.soft_ball_moment(0.5, 0.02, 2.0) == pytest.approx(L.soft_ball_trace(0.5, 0.02), rel=1e-5)
    assert L.soft_ball_moment(0.5, 1e-5, 6.0) == pytest.approx(3 * 0.5 ** 6 / 9, rel=1e-3)
    default, extent = DataFreeConfig(), DataFreeConfig(pose_power=6.0)
    assert default.pose_power == 2.0                              # the mass rule is the default (p = 6 does not train)
    assert L.pose_size_target(extent) == pytest.approx(L.soft_ball_moment(0.5, default.soft_tau, 6.0) ** (1 / 6))
    assert L.pose_size_target(default) == pytest.approx(default.cov_trace_target ** 0.5)
    # a config saved before the field existed was trained with the mass-based rule
    old = {k: v for k, v in extent.to_dict().items() if k != "pose_power"}
    assert DataFreeConfig.from_dict(old).pose_power == 2.0
    assert DataFreeConfig.from_dict(extent.to_dict()).pose_power == 6.0
    with pytest.raises(ValueError):
        DataFreeConfig(pose_power=1.0)


def test_extent_pose_terms_report_centre_and_size_errors():
    cfg = DataFreeConfig(pose_power=6.0)
    # shifted sphere: the offset of the centre (one Newton step reads a 0.05 shift about 7% short)
    mean, error = _pose_term_means(sphere(center=(0.05, 0.0, 0.0)), cfg)
    assert mean["pose_centroid"] == pytest.approx(0.05 ** 2, rel=0.25)
    assert abs(mean["pose_axes"]) < 1e-5
    # larger and smaller spheres: only the size term, the squared error of size^2
    target = L.pose_size_target(cfg) ** 2
    for radius in (1.2 * R, 0.6 * R):
        truth = (L.soft_ball_moment(radius, cfg.soft_tau, 6.0) ** (1 / 3) - target) ** 2
        mean, error = _pose_term_means(sphere(radius=radius), cfg)
        assert mean["pose_scale"] == pytest.approx(truth, rel=0.1), radius
        assert abs(mean["pose_centroid"]) < 4 * error["pose_centroid"] + 2e-5


def test_extent_size_term_pulls_a_small_shape_back_as_hard_as_the_mass_rule():
    # the pull on a shape that is too small must not fade as it shrinks (it did when the term was
    # linear in the 6th moment, and half the shapes shrank to nothing)
    def pull(power, radius):
        cfg_ = DataFreeConfig(pose_power=power)
        fb = make_fb(sphere(radius=radius), box_points(8, 40_000, seed=2))
        (grad,) = torch.autograd.grad(L.pose_terms(fb, cfg_)["pose_scale"], fb.f)
        # the soft surface layer: the extent rule acts mostly on its outer part, the mass rule on all of it
        near = fb.f.detach().abs() < 0.08
        return float(-(grad * near).sum())                        # > 0: raising the field there lowers the term

    pulls = {radius: (pull(6.0, radius), pull(2.0, radius)) for radius in (0.8 * R, 0.5 * R, 0.3 * R)}
    print("pull back on a too-small ball, extent rule and mass rule:", pulls)
    for radius, (extent, mass) in pulls.items():
        assert extent > 0 and mass > 0, radius                    # both say: grow
        assert extent > 0.3 * mass, radius                        # and the extent rule pulls comparably hard


def test_extent_rule_sizes_a_ball_with_a_rod_by_its_reach():
    # a small ball with a long thin rod: by mass it is hardly bigger than the ball, by extent it is
    def lollipop(x):
        ball = 0.3 - x.norm(dim=-1)
        along = x[..., 0].clamp(0.0, 0.8)
        rod = 0.06 - (x - torch.stack([along, torch.zeros_like(along), torch.zeros_like(along)], -1)).norm(dim=-1)
        return torch.maximum(ball, rod)

    fb = make_fb(lollipop, box_points(1, 400_000, seed=3))
    weight = L.soft_inside(fb, DataFreeConfig())
    offset2, moment2 = L.extent_pose(weight, fb.x, 2.0)
    offset6, moment6 = L.extent_pose(weight, fb.x, 6.0)
    size2, size6 = float(moment2[0]) ** 0.5, float(moment6[0]) ** (1 / 6)
    assert size6 > 1.5 * size2                                    # the rod counts
    assert float(offset6[0, 0]) > 3 * float(offset2[0, 0]) > 0    # and pulls the centre towards itself
    _, centroid, _ = L.moments(fb, DataFreeConfig())
    assert torch.allclose(offset2, centroid, atol=1e-5)           # power 2 is the mass centre


def test_pose_terms_report_the_true_squared_error():
    cfg = DataFreeConfig(pose_power=2.0)                          # the mass-based rule
    # shifted sphere: only the centroid term, equal to the squared offset
    mean, error = _pose_term_means(sphere(center=(0.05, 0.0, 0.0)), cfg)
    assert mean["pose_centroid"] == pytest.approx(0.05 ** 2, abs=4 * error["pose_centroid"] + 1e-4)
    assert abs(mean["pose_scale"]) < 1e-5 and abs(mean["pose_axes"]) < 1e-5
    # larger sphere: only the scale term
    radius = 1.2 * R
    truth = (L.soft_ball_trace(radius, cfg.soft_tau) - cfg.cov_trace_target) ** 2
    mean, error = _pose_term_means(sphere(radius=radius), cfg)
    assert mean["pose_scale"] == pytest.approx(truth, rel=0.05)
    assert abs(mean["pose_centroid"]) < 4 * error["pose_centroid"] + 2e-6
    assert abs(mean["pose_axes"]) < 1e-5
    # rotated ellipsoid: off-diagonals (and its different trace), centroid stays zero
    angle = 0.6
    rot = torch.tensor([[math.cos(angle), -math.sin(angle), 0.0],
                        [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    axes = torch.tensor([0.65, 0.5, 0.4])

    def ellipsoid(x):
        return (1.0 - ((x @ rot) / axes).norm(dim=-1)) * 0.4

    big = make_fb(ellipsoid, box_points(2, 400_000, seed=11))
    _, _, cov = L.moments(big, cfg)
    diagonal = torch.diagonal(cov, dim1=-2, dim2=-1)
    true_axes = float((cov.square().sum(dim=(-2, -1)) - diagonal.square().sum(dim=-1)).mean())
    assert true_axes > 4e-4
    mean, error = _pose_term_means(ellipsoid, cfg)
    assert mean["pose_axes"] == pytest.approx(true_axes, rel=0.08)
    assert abs(mean["pose_centroid"]) < 4 * error["pose_centroid"] + 2e-6


def test_pose_terms_have_gradients_and_ignore_the_origin_row(cfg):
    fb = make_fb(sphere(center=(0.1, 0.0, 0.0), radius=0.6), box_points(2, 1024))
    total = sum(L.pose_terms(fb, cfg).values())
    (grad,) = torch.autograd.grad(total, fb.f)
    assert torch.isfinite(grad).all() and float(grad.abs().sum()) > 0
    assert float(grad[:, 0].abs().max()) == 0.0   # row 0 (the origin) is not a random sample


def test_constraints_are_nonnegative_and_fire_when_violated(cfg):
    x = box_points(3, 4096, seed=1)
    cases = {
        "eikonal": (lambda fb: L.eikonal(fb), make_fb(sphere(gain=2.0), x), 0.5),
        "consistency": (lambda fb: L.consistency(fb), make_fb(sphere(gain=2.0), x), 0.05),
        "barrier": (lambda fb: L.barrier(fb, cfg), make_fb(sphere(gain=0.2), x), 1e-3),
        "origin_anchor": (lambda fb: L.origin_anchor(fb, cfg), make_fb(sphere(center=(0.7, 0, 0)), x), 1e-3),
        "min_volume": (lambda fb: L.min_volume(fb, cfg), make_fb(sphere(radius=0.05), x), 1e-6),
        "pose_shift": (lambda fb: L.pose(fb, cfg), make_fb(sphere(center=(0.3, 0, 0)), x), 0.05),
        "pose_size": (lambda fb: L.pose(fb, cfg), make_fb(sphere(radius=0.8), x), 0.01),
    }
    for name, (fn, fb, floor) in cases.items():
        value = float(fn(fb))
        assert value > floor, (name, value)
    # Every term is non-negative on an arbitrary smooth field.
    wobble = lambda p: 0.3 * torch.sin(3.0 * p[..., 0]) * torch.cos(2.0 * p[..., 1]) - 0.4 * p[..., 2] + 0.1
    fb = make_fb(wobble, x)
    for fn in (L.eikonal, L.consistency):
        assert float(fn(fb)) >= 0.0
    for fn in (L.barrier, L.origin_anchor, L.min_volume, L.pose):
        assert float(fn(fb, cfg)) >= 0.0


def test_eikonal_and_consistency_ignore_only_the_origin_row(cfg):
    fb = make_fb(sphere(), box_points(2, 512, seed=2))
    # The origin row of the analytic sphere has zero gradient and fp == R.
    assert float(fb.g[:, 0].norm()) < 1e-6
    assert float(fb.fp[:, 0].min()) == pytest.approx(R, abs=1e-6)
    assert float(L.eikonal(fb)) < 1e-10
    assert float(L.consistency(fb)) < 1e-6


def test_barrier_counts_only_the_band(cfg):
    x = box_points(1, 20_000, seed=4)
    fb = make_fb(sphere(gain=0.2), x)
    near = fb.f.detach().abs() < cfg.min_feature / 2
    assert 100 < int(near.sum()) < near.numel() // 4
    # the mean is over the band samples only: every one of them has |g| = 0.2
    assert float(L.barrier(fb, cfg)) == pytest.approx((cfg.barrier_floor - 0.2) ** 2, rel=1e-4)
    # a violation confined to a small part of the band is not diluted by the
    # samples far from the surface
    g = fb.g.clone()
    index = near[0].nonzero().reshape(-1)
    g[0, index[: len(index) // 10]] *= 0.0
    g[0, index[len(index) // 10:]] *= 5.0       # |g| = 1 on the rest of the band
    local = make_fb(sphere(gain=0.2), x)
    local.g = g
    share = (len(index) // 10) / len(index)
    assert float(L.barrier(local, cfg)) == pytest.approx(share * cfg.barrier_floor ** 2, rel=1e-3)
    assert float(L.barrier(local, cfg)) > 100 * cfg.tolerances["barrier"]
    # no sample near the surface: exactly zero, with a gradient path
    far = make_fb(lambda p: sphere()(p) - 5.0, x)
    assert float(L.barrier(far, cfg)) == 0.0


# -------------------------------------------------------------------- moments
def test_moments_of_sphere(cfg):
    fb = make_fb(sphere(), box_points(1, 400_000, seed=5))
    fraction, centroid, cov = L.moments(fb, cfg)
    assert fraction.shape == (1,) and centroid.shape == (1, 3) and cov.shape == (1, 3, 3)
    exact = (4.0 / 3.0) * math.pi * R**3 / 8.0
    assert float(fraction) == pytest.approx(exact, rel=0.03)
    assert float(centroid.abs().max()) < 5e-3
    assert torch.allclose(cov[0], torch.eye(3) * R * R / 5, atol=3e-3)
    assert torch.equal(L.soft_inside(fb, cfg), torch.sigmoid(fb.f / cfg.soft_tau))


def test_moments_and_descriptors_safe_on_empty_shape(cfg):
    x = box_points(2, 1024, seed=6)
    f = torch.full((2, 1024), -1.0, requires_grad=True)
    fb = fb_from_values(f, x, unit_codes(2), torch.arange(2))
    fraction, centroid, cov = L.moments(fb, cfg)
    stats = L.descriptors(fb, cfg)
    for t in (fraction, centroid, cov, stats):
        assert torch.isfinite(t).all()
    total = L.pose(fb, cfg) + stats.sum() + L.min_volume(fb, cfg)
    (grad,) = torch.autograd.grad(total, f)
    assert torch.isfinite(grad).all()


def test_moments_differentiable(cfg):
    x = box_points(2, 2048, seed=7)
    f = sphere()(x).detach().requires_grad_(True)
    fb = fb_from_values(f, x, unit_codes(2), torch.arange(2))
    (grad,) = torch.autograd.grad(L.pose(fb, cfg), f)
    assert torch.isfinite(grad).all() and float(grad.abs().sum()) > 0


# ---------------------------------------------------------------- descriptors
def _describe(field_fn, count=400_000, seed=8, cfg=None):
    fb = make_fb(field_fn, box_points(1, count, seed=seed))
    return dict(zip(L.DESCRIPTOR_NAMES, L.descriptors(fb, cfg)[0].tolist()))


def test_descriptor_tables(cfg):
    assert len(L.DESCRIPTOR_NAMES) == len(L.DESCRIPTOR_SCALES) == 10
    assert all(s > 0 for s in L.DESCRIPTOR_SCALES)
    fb = make_fb(sphere(), box_points(3, 512, seed=9))
    assert L.descriptors(fb, cfg).shape == (3, len(L.DESCRIPTOR_NAMES))


def test_descriptors_on_sphere(cfg):
    d = _describe(sphere(), cfg=cfg)
    print("sphere R=0.5:", {k: round(v, 4) for k, v in d.items()})
    assert d["volume_fraction"] == pytest.approx((4 / 3) * math.pi * R**3 / 8, rel=0.03)
    assert d["surface_area"] == pytest.approx(4 * math.pi * R * R, rel=0.04)
    assert d["sphericity"] == pytest.approx(1.0, abs=0.05)
    assert d["elongation"] == pytest.approx(1.0, abs=0.03)
    assert d["flatness"] == pytest.approx(1.0, abs=0.03)
    assert d["mean_depth"] == pytest.approx(0.25, abs=0.02)
    s = cfg.min_feature
    assert d["offset_growth"] == pytest.approx(1 + s / R + s * s / (3 * R * R), abs=0.06)
    # thickness profile of a ball: depth at level q is R (1 - (1 - q)^(1/3))
    for name, q in (("depth_p25", 0.25), ("depth_p50", 0.5), ("depth_p90", 0.9)):
        assert d[name] == pytest.approx(1 - (1 - q) ** (1 / 3), abs=0.02)


def test_thickness_profile_separates_thin_limbs_from_a_fat_body(cfg):
    """Same reach, different limb thickness: the profile differs, and it has a gradient."""
    def limbed(radius):
        core = sphere(radius=0.3)

        def field(p):
            d = p - torch.tensor([0.0, 0.0, 0.0]).to(p)
            along = d[..., 0].clamp(0.0, 0.7)                     # a limb along +x out to 0.7
            limb = radius - torch.sqrt((d[..., 0] - along) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2 + 1e-12)
            return torch.maximum(core(p), limb)
        return field

    thin, fat = _describe(limbed(0.05), cfg=cfg), _describe(limbed(0.15), cfg=cfg)
    print("thin limb:", {k: round(thin[k], 3) for k in ("depth_p25", "depth_p50", "depth_p90")},
          "fat limb:", {k: round(fat[k], 3) for k in ("depth_p25", "depth_p50", "depth_p90")})
    gaps = [abs(thin[k] - fat[k]) / s for k, s in zip(L.DESCRIPTOR_NAMES[-3:], L.DESCRIPTOR_SCALES[-3:])]
    assert max(gaps) > 0.5                                        # at least half a unit on one level
    fb = make_fb(limbed(0.1), box_points(1, 20_000, seed=2))
    f = fb.f.detach().clone().requires_grad_(True)
    out = L.descriptors(dataclasses.replace(fb, f=f), cfg)[0, -3:].sum()
    (grad,) = torch.autograd.grad(out, f)
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0


def test_descriptors_on_elongated_and_flat_shapes(cfg):
    prolate = _describe(ellipsoid(0.8, 0.4, 0.4), cfg=cfg)
    oblate = _describe(ellipsoid(0.7, 0.7, 0.3), cfg=cfg)
    ball = _describe(sphere(), cfg=cfg)
    print("prolate 0.8/0.4/0.4:", {k: round(v, 4) for k, v in prolate.items()})
    print("oblate 0.7/0.7/0.3:", {k: round(v, 4) for k, v in oblate.items()})
    assert prolate["elongation"] == pytest.approx(2.0, abs=0.1)
    assert prolate["flatness"] == pytest.approx(1.0, abs=0.05)
    assert oblate["elongation"] == pytest.approx(1.0, abs=0.05)
    assert oblate["flatness"] == pytest.approx(0.7 / 0.3, abs=0.12)
    # Exact sphericity of the 0.8/0.4/0.4 prolate ellipsoid is 0.929.
    assert prolate["sphericity"] == pytest.approx(0.929, abs=0.05)
    assert prolate["sphericity"] < ball["sphericity"]
    assert oblate["sphericity"] < ball["sphericity"]
    assert prolate["mean_depth"] < ball["mean_depth"]


def test_descriptors_on_two_spheres(cfg):
    d = _describe(two_spheres(), cfg=cfg)
    print("two spheres r=0.3 at x=+-0.45:", {k: round(v, 4) for k, v in d.items()})
    assert d["elongation"] > 2.0
    assert d["sphericity"] < 0.9


def test_descriptors_differentiable_wrt_f(cfg):
    x = box_points(2, 4096, seed=10)
    f = ellipsoid(0.7, 0.5, 0.3)(x).detach().requires_grad_(True)
    fb = fb_from_values(f, x, unit_codes(2), torch.arange(2))
    stats = L.descriptors(fb, cfg)
    for k, name in enumerate(L.DESCRIPTOR_NAMES):
        (grad,) = torch.autograd.grad(stats[:, k].sum(), f, retain_graph=True)
        assert torch.isfinite(grad).all(), name
        assert float(grad.abs().sum()) > 0, name
    # Degenerate covariance (exact sphere): eigenvalue ratios still give finite gradients.
    f2 = sphere()(x).detach().requires_grad_(True)
    stats2 = L.descriptors(fb_from_values(f2, x, unit_codes(2), torch.arange(2)), cfg)
    (grad2,) = torch.autograd.grad(stats2.sum(), f2)
    assert torch.isfinite(grad2).all()


# ------------------------------------------------------------------ diversity
def _cluster_batch(count=1024, seed=11):
    """Two clusters of three codes, shared points inside each cluster."""
    gen = torch.Generator().manual_seed(seed)
    codes = torch.cat([sample_cluster(sample_sphere(1, 32, generator=gen)[0], 3, 0.5, generator=gen) for _ in range(2)])
    cluster = torch.tensor([0, 0, 0, 1, 1, 1])
    x = box_points(2, count, seed=seed)[cluster]
    return codes, cluster, x


def test_diversity_dead_zone_is_exactly_zero(cfg):
    cfg.diversity_pose_invariant = False          # this test plants exact raw field distances
    codes, cluster, x = _cluster_batch()
    base = sphere()(x[:1]).detach()
    target = cfg.diversity_slope * chord(codes[0], codes[1])
    for factor, inside in ((1.0, True), (1.0 + 0.8 * cfg.diversity_tolerance, True),
                           (1.0 - 0.8 * cfg.diversity_tolerance, True),
                           (1.0 + 1.5 * cfg.diversity_tolerance, False),
                           (1.0 - 1.5 * cfg.diversity_tolerance, False)):
        f = torch.cat([base, base + factor * target]).requires_grad_(True)
        fb = fb_from_values(f, x[:2], codes[:2], cluster[:2])
        loss, info = L.diversity(fb, cfg, step=0)
        assert float(info["pairs"]) == 1
        assert float(info["mean_ratio"]) == pytest.approx(factor, rel=1e-4)
        if inside:
            assert float(loss) == 0.0
            assert float(info["dead_zone_fraction"]) == 1.0
            (grad,) = torch.autograd.grad(loss, f)
            assert float(grad.abs().max()) == 0.0
        else:
            expected = (abs(factor - 1.0) - cfg.diversity_tolerance) ** cfg.diversity_power
            assert float(loss) == pytest.approx(expected, rel=1e-3)
            assert float(info["dead_zone_fraction"]) == 0.0


def test_diversity_uses_only_pairs_within_a_cluster(cfg):
    codes, cluster, x = _cluster_batch()
    f = torch.randn(6, x.shape[1], generator=torch.Generator().manual_seed(0)) * 0.05
    fb = fb_from_values(f.clone().requires_grad_(True), x, codes, cluster)
    loss, info = L.diversity(fb, cfg, step=0)
    assert float(info["pairs"]) == 6  # 2 clusters x C(3, 2)
    # Changing cluster 1 leaves the per-pair terms of cluster 0 untouched.
    fb0 = fb_from_values(f[:3].clone(), x[:3], codes[:3], cluster[:3])
    fb1 = fb_from_values(f[3:].clone(), x[3:], codes[3:], cluster[3:])
    l0, _ = L.diversity(fb0, cfg, step=0)
    l1, _ = L.diversity(fb1, cfg, step=0)
    assert float(loss) == pytest.approx(0.5 * (float(l0) + float(l1)), rel=1e-5)


def test_diversity_ignores_points_beyond_bound_radius(cfg):
    codes, cluster, x = _cluster_batch()
    f = sphere()(x[:2]).detach().clone()
    f[1] += 0.03
    far = x[0].norm(dim=-1) >= cfg.bound_radius
    assert far.any()
    noisy = f.clone()
    noisy[1, far] += 5.0
    a, _ = L.diversity(fb_from_values(f, x[:2], codes[:2], cluster[:2]), cfg, step=0)
    b, _ = L.diversity(fb_from_values(noisy, x[:2], codes[:2], cluster[:2]), cfg, step=0)
    assert float(a) == float(b)


def test_diversity_single_code_clusters(cfg):
    codes, _, x = _cluster_batch()
    # One cluster of three and three singletons: only the three pairs count.
    cluster = torch.tensor([0, 0, 0, 1, 2, 3])
    f = (sphere()(x).detach() + 0.01 * torch.arange(6).unsqueeze(1)).requires_grad_(True)
    loss, info = L.diversity(fb_from_values(f, x, codes, cluster), cfg, step=0)
    assert float(info["pairs"]) == 3
    assert torch.isfinite(loss)
    # No pair at all: a zero that still back-propagates.
    f2 = sphere()(x).detach().requires_grad_(True)
    loss2, info2 = L.diversity(fb_from_values(f2, x, codes, torch.arange(6)), cfg, step=10**9)
    assert float(loss2) == 0.0 and float(info2["pairs"]) == 0
    (grad,) = torch.autograd.grad(loss2, f2)
    assert torch.isfinite(grad).all()
    assert all(torch.isfinite(v).all() for v in info2.values())


@pytest.mark.parametrize("step", [0, 10**9])
def test_diversity_identical_fields_are_finite(cfg, step):
    codes, cluster, x = _cluster_batch()
    f = sphere()(x).detach().requires_grad_(True)  # same field for every code: d == 0
    loss, info = L.diversity(fb_from_values(f, x, codes, cluster), cfg, step=step)
    expected = (1.0 - cfg.diversity_tolerance) ** cfg.diversity_power
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    assert float(info["mean_ratio"]) == 0.0
    (grad,) = torch.autograd.grad(loss, f)
    assert torch.isfinite(grad).all()
    # Duplicate codes (chord == 0) must not divide by zero either.
    same_codes = codes[:1].expand(6, -1).clone()
    f3 = (sphere()(x).detach() + 0.01 * torch.arange(6).unsqueeze(1)).requires_grad_(True)
    loss3, _ = L.diversity(fb_from_values(f3, x, same_codes, cluster), cfg, step=step)
    (grad3,) = torch.autograd.grad(loss3, f3)
    assert torch.isfinite(loss3) and torch.isfinite(grad3).all()


def test_diversity_phase2_adds_descriptor_distance(cfg):
    codes, cluster, x = _cluster_batch(count=8192)
    fields = [sphere(), ellipsoid(0.8, 0.4, 0.4), ellipsoid(0.7, 0.7, 0.3)]
    f = torch.cat([fn(x[i : i + 1]) for i, fn in enumerate(fields)]).detach().requires_grad_(True)
    fb = fb_from_values(f, x[:3], codes[:3], cluster[:3])
    _, before = L.diversity(fb, cfg, step=cfg.phase2_start_step - 1)
    loss, after = L.diversity(fb, cfg, step=cfg.phase2_start_step)
    assert float(before["phase2"]) == 0.0 and float(after["phase2"]) == 1.0
    assert float(after["mean_shape_distance"]) > float(before["mean_shape_distance"])
    (grad,) = torch.autograd.grad(loss, f)
    assert torch.isfinite(grad).all()


# ---------------------------------------------------------------- multipliers
def _mult(**kw):
    args = dict(tolerances={"a": 0.1, "b": 0.1}, mult_init=1.0, mult_min=1e-2, mult_max=1e4,
                mult_rate=0.02, mult_ema=0.95)
    args.update(kw)
    return AdaptiveMultipliers(**args)


def test_multipliers_rise_and_decay():
    m = _mult()
    for _ in range(1000):
        m.update({"a": 1.0, "b": 0.0})
    assert m.multiplier("a") == 1e4
    assert m.multiplier("b") == 1e-2
    # One step moves by exactly the documented factors.
    m2 = _mult()
    m2.update({"a": 1.0, "b": 0.0})
    assert m2.multiplier("a") == pytest.approx(1.02)
    assert m2.multiplier("b") == pytest.approx(0.99)
    # A once-violated constraint relaxes after it is met (the EMA has to fall first).
    for _ in range(2000):
        m.update({"a": 0.0})
    assert m.multiplier("a") == 1e-2


def test_multipliers_use_the_ema_not_the_last_value():
    m = _mult()
    for _ in range(50):
        m.update({"a": 1.0})
    before = m.multiplier("a")
    m.update({"a": 0.0})  # EMA is still 0.95 > 0.1
    assert m.multiplier("a") == pytest.approx(before * 1.02)
    assert m.ema("a") == pytest.approx(0.95)
    m.update({"a": float("nan")})
    assert m.multiplier("a") == pytest.approx(before * 1.02) and m.ema("a") == pytest.approx(0.95)


def test_multipliers_combine_has_no_gradient_through_lambda():
    m = _mult(mult_init=3.0)
    objective = torch.tensor(2.0, requires_grad=True)
    ca = torch.tensor(0.5, requires_grad=True)
    cnew = torch.tensor(0.25, requires_grad=True)
    total = m.combine(objective, {"a": ca, "brand_new": cnew})
    assert m.multiplier("brand_new") == 3.0  # created on first use at mult_init
    assert float(total) == pytest.approx(2.0 + 3.0 * 0.5 + 3.0 * 0.25)
    total.backward()
    assert float(objective.grad) == 1.0
    assert float(ca.grad) == 3.0 and float(cnew.grad) == 3.0
    assert all(isinstance(v, float) for v in m.multipliers().values())


def test_multipliers_unknown_names_and_config(cfg):
    m = AdaptiveMultipliers(cfg)
    assert m.multiplier("never_seen") == cfg.mult_init
    m.update({"also_new": 0.0})
    assert "also_new" in m.multipliers()
    assert m.mult_max == cfg.mult_max and m.mult_rate == cfg.mult_rate
    m.update({"tensor_value": torch.tensor(1.0)})
    assert isinstance(m.ema("tensor_value"), float)


def test_multipliers_state_round_trip():
    m = _mult()
    for i in range(37):
        m.update({"a": 1.0, "b": 0.0, "c": 0.3 * (i % 2)})
    state = m.state_dict()
    other = _mult()
    other.load_state_dict(state)
    assert other.multipliers() == m.multipliers()
    assert other.emas() == m.emas()
    for _ in range(5):
        m.update({"a": 0.05, "b": 0.5, "c": 0.0})
        other.update({"a": 0.05, "b": 0.5, "c": 0.0})
    assert other.multipliers() == m.multipliers()
    import pickle

    assert pickle.loads(pickle.dumps(state)) == state


# ----------------------------------------------- through the real decoder
def _small_cfg():
    return DataFreeConfig(width=64, layers=4, skip_layer=2, mapping_width=32, points=512,
                          clusters=3, codes_per_cluster=3, grad_change_points=64)


def _real_batch(cfg, device, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    decoder = ModulatedSphereDecoder(cfg).to(device)
    centers = sample_sphere(cfg.clusters, cfg.latent_dim, generator=gen)
    codes = torch.cat([sample_cluster(c, cfg.codes_per_cluster, 0.5, generator=gen) for c in centers]).to(device)
    cluster = torch.arange(cfg.clusters).repeat_interleave(cfg.codes_per_cluster).to(device)
    points = sample_points(cfg, cfg.clusters, gen, device)
    return decoder, codes, cluster, points[cluster]


def _all_losses(decoder, fb, cfg, step):
    gen = torch.Generator().manual_seed(1)
    return {
        "eikonal": L.eikonal(fb),
        "consistency": L.consistency(fb),
        "grad_change": L.grad_change(decoder, fb, cfg, gen),
        "barrier": L.barrier(fb, cfg),
        "origin_anchor": L.origin_anchor(fb, cfg),
        "min_volume": L.min_volume(fb, cfg),
        "pose": L.pose(fb, cfg),
        "descriptors": L.descriptors(fb, cfg).sum(),
        "diversity": L.diversity(fb, cfg, step)[0],
        "diversity_phase2": L.diversity(fb, cfg, cfg.phase2_start_step)[0],
    }


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("perturb", [False, True])
def test_real_decoder_every_loss_backpropagates(device, perturb):
    cfg = _small_cfg()
    decoder, codes, cluster, x = _real_batch(cfg, device)
    if perturb:
        # Move off the initial sphere so every hinge is active and codes differ.
        with torch.no_grad():
            for p in decoder.parameters():
                p.add_(0.05 * torch.randn_like(p))
    fb = evaluate_field(decoder, codes, cluster, x, cfg, create_graph=True)
    assert fb.f.shape == (9, cfg.points) and fb.g.shape == (9, cfg.points, 3)
    params = [p for p in decoder.parameters() if p.requires_grad]
    terms = _all_losses(decoder, fb, cfg, step=0)
    for name, value in terms.items():
        assert value.dim() == 0 and torch.isfinite(value), name
        if name != "descriptors":
            assert float(value) >= 0.0, name
        if not value.requires_grad:
            continue
        grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        for grad in grads:
            if grad is not None:
                assert torch.isfinite(grad).all(), name
    if perturb:
        # Away from the sphere these all carry a real signal to the decoder.
        for name in ("eikonal", "consistency", "grad_change", "pose", "descriptors", "diversity"):
            grads = torch.autograd.grad(terms[name], params, retain_graph=True, allow_unused=True)
            assert sum(float(g.abs().sum()) for g in grads if g is not None) > 0, name
    assert float(L.consistency_error(fb)) >= 0.0 and not L.consistency_error(fb).requires_grad


def test_real_decoder_at_init_is_a_feasible_sphere():
    cfg = DataFreeConfig(width=64, layers=4, skip_layer=2, mapping_width=32, points=4096,
                         clusters=2, codes_per_cluster=2)
    decoder, codes, cluster, x = _real_batch(cfg, "cpu")
    fb = evaluate_field(decoder, codes, cluster, x, cfg, create_graph=True)
    values = {
        "eikonal": float(L.eikonal(fb)),
        "consistency": float(L.consistency(fb)),
        "barrier": float(L.barrier(fb, cfg)),
        "origin_anchor": float(L.origin_anchor(fb, cfg)),
        "min_volume": float(L.min_volume(fb, cfg)),
        "pose": float(L.pose(fb, cfg)),
    }
    print("real decoder at init:", {k: f"{v:.3e}" for k, v in values.items()})
    assert values["eikonal"] < 1e-3
    assert values["consistency"] < 1e-2
    assert values["barrier"] == 0.0
    assert values["origin_anchor"] == 0.0
    assert values["min_volume"] == 0.0
    assert values["pose"] < POSE_TOLERANCE_4096


def test_grad_change_accepts_prepared_modulation_and_is_small_on_sphere():
    cfg = _small_cfg()
    decoder, codes, cluster, x = _real_batch(cfg, "cpu")
    fb = evaluate_field(decoder, codes, cluster, x, cfg, create_graph=True)
    mod = decoder.prepare(codes)
    a = L.grad_change(decoder, fb, cfg, torch.Generator().manual_seed(5), mod=mod)
    b = L.grad_change(decoder, fb, cfg, torch.Generator().manual_seed(5))
    assert float(a) == pytest.approx(float(b), rel=1e-5, abs=1e-8)
    c = L.grad_change(decoder, fb, cfg)  # no generator
    assert torch.isfinite(c) and float(c) >= 0.0
    # On a sphere the normal turns by about step/|x|, so 1 - cos is small but not zero.
    assert 0.0 < float(a) < 0.05


def test_multipliers_with_real_losses():
    cfg = _small_cfg()
    decoder, codes, cluster, x = _real_batch(cfg, "cpu")
    fb = evaluate_field(decoder, codes, cluster, x, cfg, create_graph=True)
    terms = _all_losses(decoder, fb, cfg, step=0)
    objective = terms.pop("diversity")
    terms.pop("descriptors"), terms.pop("diversity_phase2")
    m = AdaptiveMultipliers(cfg)
    total = m.combine(objective, terms)
    expected = float(objective) + sum(m.multiplier(k) * float(v) for k, v in terms.items())
    assert float(total) == pytest.approx(expected, rel=1e-5)
    total.backward()
    m.update({k: float(v) for k, v in terms.items()})
    assert set(terms) <= set(m.multipliers())


# ------------------------------------------- per-constraint ranges and rates
def test_multipliers_have_their_own_ranges_and_rates():
    mult = AdaptiveMultipliers(
        tolerances={"a": 0.0, "b": 0.0, "c": 1.0}, mult_init=1.0, mult_min=1.0, mult_max=8.0,
        mult_rate=1.0, mult_ema=0.0,
        mult_ranges={"a": (1.0, 100.0), "c": (0.25, 3.0)}, mult_rates={"a": 3.0},
    )
    assert mult.range("a") == (1.0, 100.0) and mult.range("b") == (1.0, 8.0)
    assert mult.rate("a") == 3.0 and mult.rate("b") == 1.0
    for _ in range(4):
        mult.update({"a": 1.0, "b": 1.0, "c": 0.0})
    assert mult.multiplier("a") == 100.0          # x4 per step, own cap
    assert mult.multiplier("b") == 8.0            # x2 per step, global cap
    assert mult.multiplier("c") == 0.25           # x0.5 per step, own floor below the global one
    # a new name starts inside its own range
    high = AdaptiveMultipliers(mult_init=1.0, mult_ranges={"z": (5.0, 9.0)})
    assert high.multiplier("z") == 5.0
    with pytest.raises(ValueError):
        AdaptiveMultipliers(mult_ranges={"z": (9.0, 5.0)})


def test_multiplier_ranges_survive_a_state_dict_round_trip():
    kwargs = dict(tolerances={"a": 0.0}, mult_init=1.0, mult_min=1.0, mult_max=8.0, mult_rate=1.0,
                  mult_ema=0.0)
    mult = AdaptiveMultipliers(mult_ranges={"a": (1.0, 100.0)}, mult_rates={"a": 3.0}, **kwargs)
    for _ in range(3):
        mult.update({"a": 1.0, "b": 1.0})
    state = mult.state_dict()
    assert state["mult_ranges"] == {"a": (1.0, 100.0)} and state["mult_rates"] == {"a": 3.0}
    # same ranges: values come back as saved
    same = AdaptiveMultipliers(mult_ranges={"a": (1.0, 100.0)}, mult_rates={"a": 3.0}, **kwargs)
    same.load_state_dict(state)
    assert same.multipliers() == mult.multipliers() == {"a": 64.0, "b": 8.0}
    # a config with a tighter range wins over the saved value ...
    tight = AdaptiveMultipliers(mult_ranges={"a": (1.0, 10.0)}, **kwargs)
    tight.load_state_dict(state)
    assert tight.multiplier("a") == 10.0 and tight.rate("a") == 1.0
    # ... unless the saved hyperparameters are asked for
    tight.load_state_dict(state, load_hyperparameters=True)
    assert tight.multiplier("a") == 64.0 and tight.rate("a") == 3.0 and tight.range("a") == (1.0, 100.0)


def test_multipliers_read_ranges_and_rates_from_the_config():
    cfg = DataFreeConfig()
    mult = AdaptiveMultipliers(cfg)
    for name in ("bridge", "delete", "fill", "widen", "thicken", "barrier"):
        cap = 1e3 if name in ("widen", "thicken") else 1e4
        assert mult.range(name) == (1.0, cap) and mult.rate(name) == 0.05
    assert mult.range("pose_centroid") == (1.0, 50.0)
    assert mult.range("pose_scale") == (1.0, 50.0) and mult.range("pose_axes") == (1.0, 500.0)
    assert mult.range("eikonal") == (cfg.mult_min, cfg.mult_max) == (1.0, 200.0)
    assert mult.rate("eikonal") == mult.rate("pose_scale") == cfg.mult_rate
    assert cfg.tolerances["pose_centroid"] == 2.5e-4 and cfg.tolerances["barrier"] == 1e-4
    again = DataFreeConfig.from_dict(cfg.to_dict())
    assert AdaptiveMultipliers(again).range("delete") == (1.0, 1e4)


# ------------------------------------------------------- pose-invariant diversity
def _moved(field_fn, shift=(0.0, 0.0, 0.0), angle=0.0, scale=1.0):
    """The same shape translated, turned about z and uniformly resized."""
    c, s_ = math.cos(angle), math.sin(angle)
    rot = torch.tensor([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    t = torch.tensor(shift)

    def moved(p):
        q = ((p - t.to(p)) @ rot.to(p)) / scale       # world point -> body frame
        return scale * field_fn(q)
    return moved


def _smooth_blob(p):
    """A smooth, non-round test shape (no kink or singular point anywhere), positive inside."""
    scale = torch.tensor([1.0, 1.5, 2.0]).to(p)
    return 0.5 - torch.sqrt((p * scale).square().sum(-1) + 0.01)


def _pair_fb(field_a, field_b, count=4096, seed=3):
    x = box_points(1, count, seed).expand(2, -1, -1).clone()
    codes = unit_codes(2, seed=seed)
    cluster = torch.zeros(2, dtype=torch.long)
    fa = make_fb(field_a, x[:1], codes[:1], cluster[:1])
    fb_ = make_fb(field_b, x[1:], codes[1:], cluster[1:])
    return FieldBatch(codes=codes, cluster=cluster, x=x, f=torch.cat([fa.f, fb_.f]),
                      g=torch.cat([fa.g, fb_.g]), xp=torch.cat([fa.xp, fb_.xp]),
                      fp=torch.cat([fa.fp, fb_.fp]))


@pytest.mark.parametrize("move", [dict(shift=(0.02, -0.01, 0.015)), dict(angle=0.05), dict(scale=1.04),
                                  dict(shift=(0.01, 0.01, 0.0), angle=0.03, scale=0.97)])
def test_pose_invariant_diversity_ignores_small_rigid_motion_and_resize(cfg, move):
    base = _smooth_blob
    fb = _pair_fb(base, _moved(base, **move))
    cfg.diversity_pose_invariant = False
    _, raw = L.diversity(fb, cfg)
    cfg.diversity_pose_invariant = True
    _, inv = L.diversity(fb, cfg)
    assert float(raw["mean_shape_distance"]) > 2e-3
    # the pose part is at least 90% of the raw difference for these small motions
    assert float(inv["mean_shape_distance"]) < 0.1 * float(raw["mean_shape_distance"])
    assert float(inv["pose_fraction"]) > 0.9


def test_pose_invariant_diversity_keeps_real_deformation(cfg):
    base = _smooth_blob

    def dented(p):                                    # a local dent on one side: not a pose change
        return base(p) - 0.06 * torch.exp(-((p - torch.tensor([0.45, 0.0, 0.0]).to(p)) ** 2).sum(-1) / 0.02)
    fb = _pair_fb(base, dented)
    cfg.diversity_pose_invariant = False
    _, raw = L.diversity(fb, cfg)
    cfg.diversity_pose_invariant = True
    _, inv = L.diversity(fb, cfg)
    assert float(inv["mean_shape_distance"]) > 0.6 * float(raw["mean_shape_distance"])


def test_pose_invariant_diversity_gradient_has_no_pose_component(cfg):
    base = _smooth_blob
    fb = _pair_fb(base, _moved(base, shift=(0.03, 0.0, 0.0), angle=0.04))
    f = fb.f.detach().clone().requires_grad_(True)
    fb = dataclasses.replace(fb, f=f)
    cfg.diversity_pose_invariant = True
    loss, _ = L.diversity(fb, cfg)
    (grad,) = torch.autograd.grad(loss, f)
    x, g = fb.x[0], 0.5 * (fb.g[0] + fb.g[1])
    mask = (x.square().sum(-1) < cfg.bound_radius ** 2).to(g.dtype)
    value = 0.5 * (fb.f[0] + fb.f[1]).detach()
    basis = torch.cat([g, torch.linalg.cross(x, g, dim=-1), value[:, None] - (g * x).sum(-1, keepdim=True)], -1) * mask[:, None]
    along = (basis.T @ grad[0]).abs() / (basis.norm(dim=0) * grad[0].norm() + 1e-20)
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
    assert float(along.max()) < 2e-2                  # ridge leaves a tiny remainder


def test_pose_invariant_diversity_is_safe_on_identical_spheres(cfg):
    fb = _pair_fb(sphere(), sphere())                 # rotation columns vanish on a sphere
    f = fb.f.detach().clone().requires_grad_(True)
    cfg.diversity_pose_invariant = True
    loss, info = L.diversity(dataclasses.replace(fb, f=f), cfg)
    (grad,) = torch.autograd.grad(loss, f, allow_unused=True)
    assert bool(torch.isfinite(loss)) and (grad is None or bool(torch.isfinite(grad).all()))


# ----------------------------------------------------------------------- ordinal
def _ordinal_cfg(cfg):
    cfg.diversity_pose_invariant = False
    cfg.diversity_mode = "ordinal"
    return cfg


def test_ordinal_is_zero_for_a_distance_preserving_family_above_the_floor(cfg):
    fb, _ = _linear_family(rank=32, slope=0.2)        # twice the floor: no curve is imposed
    loss, info = L.diversity(fb, _ordinal_cfg(cfg))
    assert float(info["order_violations"]) < 0.02
    assert float(loss) < 1e-3


def test_ordinal_has_no_upper_limit_or_curve(cfg):
    """Any increasing relation between latent and shape distance is accepted above the floor."""
    cfg = _ordinal_cfg(cfg)
    fb, f = _linear_family(rank=32, slope=0.2)
    base, _ = L.diversity(fb, cfg)
    bigger = dataclasses.replace(fb, f=f.detach() * 5.0)        # five times the shape change
    loss, info = L.diversity(bigger, cfg)
    assert float(loss) < 1e-3 and float(info["mean_ratio"]) > 3.0
    cfg.diversity_mode = "slope"
    band, _ = L.diversity(bigger, cfg)
    assert float(band) > 1.0                                      # the slope loss rejects it


def test_ordinal_floor_fires_when_shapes_barely_differ(cfg):
    fb, f = _linear_family(rank=32, slope=0.02)       # a fifth of the floor
    loss, info = L.diversity(dataclasses.replace(fb, f=f), _ordinal_cfg(cfg))
    assert float(info["mean_ratio"]) < 0.5 and float(loss) > 0.3
    (grad,) = torch.autograd.grad(loss, f)
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0


def test_ordinal_penalises_a_single_dial_family(cfg):
    fb, f = _linear_family(rank=1, slope=0.5)         # large changes, but all along one mode
    loss, info = L.diversity(dataclasses.replace(fb, f=f), _ordinal_cfg(cfg))
    assert float(info["order_violations"]) > 0.15     # far codes across the dial look alike
    full, full_info = L.diversity(_linear_family(rank=32, slope=0.5)[0], cfg)
    assert float(loss) > 10 * float(full) + 1e-3


def test_ordinal_is_finite_for_identical_fields(cfg):
    fb, f = _linear_family(rank=32)
    same = dataclasses.replace(fb, f=(f.detach()[:1].expand_as(f)).clone().requires_grad_(True))
    loss, _ = L.diversity(same, _ordinal_cfg(cfg))
    (grad,) = torch.autograd.grad(loss, same.f, allow_unused=True)
    assert bool(torch.isfinite(loss)) and (grad is None or bool(torch.isfinite(grad).all()))


# ------------------------------------------------------------------------- repel
def test_repel_mode_rewards_more_shape_distance_without_a_cap(cfg):
    cfg.diversity_pose_invariant = False
    cfg.diversity_mode = "repel"
    fb, f = _linear_family(rank=32, slope=0.1)
    base, info = L.diversity(fb, cfg)
    more, _ = L.diversity(dataclasses.replace(fb, f=f.detach() * 3.0), cfg)
    assert float(more) < float(base)                              # more distance, lower loss
    assert float(more) == pytest.approx(float(base) - math.log(3.0), abs=0.05)   # logarithmic
    assert float(info["order_violations"]) < 0.02                 # still reported
    (grad,) = torch.autograd.grad(base, f)
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
    same = dataclasses.replace(fb, f=(f.detach()[:1].expand_as(f)).clone().requires_grad_(True))
    assert bool(torch.isfinite(L.diversity(same, cfg)[0]))        # identical fields are finite


# ----------------------------------------------------------------------- nearest
def _global_family(dials, count=1024, members=16, seed=9, size=0.1):
    """``members`` random codes sharing points; the field varies along ``dials`` independent modes."""
    gen = torch.Generator().manual_seed(seed)
    codes = sample_sphere(members, 32, generator=gen)
    x = (box_points(1, count, seed=seed) * 0.5).expand(members, -1, -1).clone()
    modes = torch.linalg.qr(torch.randn(count, 32, generator=gen))[0] * math.sqrt(count)
    f = (size * codes[:, :dials] @ modes[:, :dials].T).requires_grad_(True)
    return fb_from_values(f, x, codes, torch.zeros(members, dtype=torch.long)), f


def test_nearest_mode_prefers_many_dials_at_equal_total_variation(cfg):
    cfg.diversity_pose_invariant = False
    cfg.diversity_mode = "nearest"
    # same total field variance, spread over 1 dial or over 16
    one, _ = _global_family(1, size=0.1 * 4.0)
    many, f = _global_family(16, size=0.1)
    loss_one, info_one = L.diversity(one, cfg)
    loss_many, info_many = L.diversity(many, cfg)
    print("nearest distance: 1 dial", float(info_one["nearest_distance"]), "16 dials", float(info_many["nearest_distance"]))
    assert float(info_many["nearest_distance"]) > 2 * float(info_one["nearest_distance"])
    assert float(loss_many) < float(loss_one) - 0.5
    (grad,) = torch.autograd.grad(loss_many, f)
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
    same = dataclasses.replace(many, f=(f.detach()[:1].expand_as(f)).clone().requires_grad_(True))
    assert bool(torch.isfinite(L.diversity(same, cfg)[0]))


def test_variety_surface_band_measures_surface_distance_not_volume():
    """Ball, ball plus a thin spike, ball grown a little. Over the whole ball the spike counts for less
    than the slight growth; near the surfaces it counts for more, and the grown ball reads its growth."""
    gen = torch.Generator().manual_seed(3)
    x = (torch.rand(1, 20000, 3, generator=gen) * 1.6 - 0.8).expand(3, -1, -1).clone()

    def fields(p):
        rho = p.norm(dim=-1)
        ball = 0.4 - rho
        along = p[..., 2].clamp(0.0, 0.75)                             # a rod of radius 0.04 along +z, tip at 0.75
        spike = 0.04 - (p - torch.stack([torch.zeros_like(along), torch.zeros_like(along), along], -1)).norm(dim=-1)
        return torch.stack([ball[0], torch.maximum(ball[1], spike[1]), 0.43 - rho[2]])

    fb = make_fb(fields, x)
    codes = unit_codes(3)
    fb = dataclasses.replace(fb, codes=codes, cluster=torch.zeros(3, dtype=torch.long))
    index = torch.arange(fb.f.shape[1])
    none = fb.f.new_zeros((0, index.numel()))

    def nearest(band):
        cfg = DataFreeConfig(diversity_mode="nearest", nearest_k=2, diversity_pose_invariant=False, variety_surface_band=band)
        f = (fb.f if getattr(fb, "f_move", None) is None else fb.f_move).detach()
        mask = (fb.x.detach().square().sum(-1) < cfg.bound_radius ** 2).float()
        loss, info = L.nearest_pool(fb, none, index, cfg)
        # the pair distances the function used, recomputed here from its documented rule
        def pair(a, b):
            w = mask[a]
            if band > 0:
                close = w * (torch.minimum(f[a].abs(), f[b].abs()) < band).float()
                w = close if float(close.sum()) >= 8 else w
            return float(((f[a] - f[b]).abs() * w).sum() / w.sum())
        assert float(info["raw_nearest_distance"]) == pytest.approx((min(pair(0, 1), pair(0, 2)) + min(pair(1, 0), pair(1, 2))
                                                                     + min(pair(2, 0), pair(2, 1))) / 3, rel=1e-4)
        return pair(0, 1), pair(0, 2), loss

    spike_all, grown_all, _ = nearest(0.0)
    spike_near, grown_near, loss = nearest(0.05)
    assert spike_all < grown_all                                        # by volume the spike barely registers
    # Near the surfaces it counts for more, though still by its share of the surface AREA (the ball's
    # own surface, where nothing changed, is most of the band), so the gain for one thin rod is modest.
    assert spike_near > 1.2 * spike_all
    assert spike_near / grown_near > 1.2 * spike_all / grown_all
    assert grown_near == pytest.approx(0.03, abs=2e-3)                  # the grown ball's surface is 0.03 away
    (grad,) = torch.autograd.grad(loss, fb.f)
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0


def test_nearest_pool_matches_in_batch_search_and_finds_closer_pool_neighbours(cfg):
    cfg.diversity_pose_invariant = False
    cfg.diversity_mode = "nearest"
    cfg.nearest_k = 15
    fb, f = _global_family(16, members=16)
    index = torch.arange(1, fb.f.shape[1])
    # with an empty pool and k = all others, it reproduces the in-batch nearest-neighbour loss
    inside, _ = L.diversity(fb, cfg)
    pooled, info = L.nearest_pool(fb, fb.f.new_zeros((0, index.numel())), index, cfg)
    assert float(pooled) == pytest.approx(float(inside), rel=0.02)
    assert float(info["pool_neighbour_share"]) == 0.0
    # a pool holding near-copies of the batch shapes supplies much closer neighbours
    cfg.nearest_k = 4
    twins = (fb.f.detach() * 1.02)[:, index]
    loss, info2 = L.nearest_pool(fb, twins, index, cfg)
    assert float(info2["nearest_distance"]) < 0.2 * float(info["nearest_distance"])
    assert float(info2["pool_neighbour_share"]) > 0.2
    (grad,) = torch.autograd.grad(loss, f)
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0


def test_nearest_pool_info_covers_the_logged_keys_and_codes_do_not_change_the_loss(cfg):
    cfg.diversity_mode = "nearest"
    cfg.nearest_k = 4
    fb, f = _global_family(16, members=16)
    index = torch.arange(1, fb.f.shape[1])
    gen = torch.Generator().manual_seed(3)
    pool_codes = sample_sphere(24, 32, generator=gen)
    pool_f = 0.05 * torch.randn(24, index.numel(), generator=gen)
    plain, info_plain = L.nearest_pool(fb, pool_f, index, cfg)
    coded, info = L.nearest_pool(fb, pool_f, index, cfg, pool_codes=pool_codes)
    assert float(coded) == pytest.approx(float(plain), abs=1e-6)
    keys = ("nearest_distance", "pool_neighbour_share", "raw_nearest_distance", "mean_shape_distance",
            "mean_latent_distance", "mean_ratio", "dead_zone_fraction", "phase2", "order_violations")
    for each in (info_plain, info):
        for key in keys:
            assert key in each and math.isfinite(float(each[key]))
            assert not (isinstance(each[key], torch.Tensor) and each[key].requires_grad)
        assert float(each["phase2"]) == 0.0 and float(each["order_violations"]) == 0.0
    assert float(info["mean_latent_distance"]) > 0.0 and float(info["mean_ratio"]) > 0.0
    assert 0.0 <= float(info["dead_zone_fraction"]) <= 1.0
    for key in ("mean_latent_distance", "mean_ratio", "dead_zone_fraction"):
        assert float(info_plain[key]) == 0.0
    for key in ("nearest_distance", "mean_shape_distance", "pool_neighbour_share"):
        assert float(info[key]) == pytest.approx(float(info_plain[key]), abs=1e-6)
    assert float(info["mean_shape_distance"]) >= float(info["nearest_distance"])
    # the ratio is distance over slope times chord: a far larger slope empties the "above the floor" share
    steep = dataclasses.replace(cfg, diversity_slope=1e6)             # ``cfg`` is shared by the module
    assert float(L.nearest_pool(fb, pool_f, index, steep, pool_codes=pool_codes)[1]["dead_zone_fraction"]) == 0.0


def test_pass_by_pass_neighbour_gradients_sum_to_the_whole_batch_gradient(cfg):
    """Each half sees the other half as fixed neighbours; the summed gradients equal the full ones."""
    cfg.diversity_pose_invariant = False
    cfg.diversity_mode = "nearest"
    cfg.nearest_k = 7
    fb, f = _global_family(16, members=8)
    index = torch.arange(1, fb.f.shape[1])
    full, _ = L.nearest_pool(fb, f.new_zeros((0, index.numel())), index, cfg)
    (grad_full,) = torch.autograd.grad(full, f)

    def half(rows, rest):
        part = FieldBatch(codes=fb.codes[rows], cluster=fb.cluster[rows], x=fb.x[rows], f=f[rows],
                          g=fb.g[rows], xp=fb.xp[rows], fp=fb.fp[rows])
        loss, info = L.nearest_pool(part, f.detach()[rest][:, index], index, cfg, pool_codes=fb.codes[rest],
                                    pool_terms=True)
        return loss, info

    first, second = torch.arange(0, 4), torch.arange(4, 8)
    loss_a, info_a = half(first, second)
    loss_b, _ = half(second, first)
    # each pass returns (sum of all 8 terms) / 4 = 2 x the whole-set mean; the trainer divides by the
    # number of passes, so the passes' gradients add up to the whole-set gradient
    assert float(loss_a) == pytest.approx(2.0 * float(full), rel=1e-5)
    assert float(loss_b) == pytest.approx(2.0 * float(full), rel=1e-5)
    (grad_sum,) = torch.autograd.grad(0.5 * (loss_a + loss_b), f)
    assert torch.allclose(grad_sum, grad_full, rtol=1e-4, atol=1e-9)
    # without the pool shapes' own terms the gradient is incomplete
    one_sided = sum(torch.autograd.grad(0.5 * L.nearest_pool(
        FieldBatch(codes=fb.codes[r], cluster=fb.cluster[r], x=fb.x[r], f=f[r], g=fb.g[r], xp=fb.xp[r], fp=fb.fp[r]),
        f.detach()[o][:, index], index, cfg)[0], f)[0] for r, o in ((first, second), (second, first)))
    assert not torch.allclose(one_sided, grad_full, rtol=1e-2, atol=1e-9)
    assert float(info_a["pool_neighbour_share"]) > 0.3                  # neighbours from the other pass are used
    # passes of unequal size: with norm = the mean size the summed gradients are still the whole-set
    # gradient; without it the small pass counts its shapes too heavily
    small, large = torch.arange(0, 1), torch.arange(1, 8)

    def uneven(rows, rest, norm):
        part = FieldBatch(codes=fb.codes[rows], cluster=fb.cluster[rows], x=fb.x[rows], f=f[rows],
                          g=fb.g[rows], xp=fb.xp[rows], fp=fb.fp[rows])
        return L.nearest_pool(part, f.detach()[rest][:, index], index, cfg, pool_terms=True, norm=norm)[0]

    (grad_even,) = torch.autograd.grad(0.5 * (uneven(small, large, 4.0) + uneven(large, small, 4.0)), f)
    assert torch.allclose(grad_even, grad_full, rtol=1e-4, atol=1e-9)
    (grad_skew,) = torch.autograd.grad(0.5 * (uneven(small, large, None) + uneven(large, small, None)), f)
    assert not torch.allclose(grad_skew, grad_full, rtol=1e-2, atol=1e-9)


# ----------------------------------------------------------------- local chamfer
def _cluster_of(fields, count=3000, seed=6):
    x = box_points(1, count, seed=seed).expand(len(fields), -1, -1).clone()
    x[:, 0] = 0.0
    codes = unit_codes(len(fields), seed=seed)
    parts = [make_fb(fn, x[i:i + 1], codes[i:i + 1], torch.zeros(1, dtype=torch.long)) for i, fn in enumerate(fields)]
    return FieldBatch(codes=codes, cluster=torch.zeros(len(fields), dtype=torch.long), x=x,
                      f=torch.cat([p.f for p in parts]), g=torch.cat([p.g for p in parts]),
                      xp=torch.cat([p.xp for p in parts]), fp=torch.cat([p.fp for p in parts]))


def _bumped(height):
    """A sphere with one local bump of the given height on its +x side."""
    base = sphere(radius=0.45)
    centre = torch.tensor([0.45, 0.0, 0.0])
    return lambda p: base(p) + height * torch.exp(-((p - centre.to(p)) ** 2).sum(-1) / 0.01)


def test_local_chamfer_measures_the_gap_between_surfaces(cfg):
    gen = torch.Generator().manual_seed(0)
    same, info_same = L.local_chamfer(_cluster_of([sphere(radius=0.45)] * 2), cfg, gen)
    apart, info = L.local_chamfer(_cluster_of([sphere(radius=0.45), sphere(radius=0.50)]), cfg, gen)
    print("identical spheres: distance", float(info_same["local_distance"]), "floor", float(info_same["local_floor"]),
          "| radii 0.45 vs 0.50: distance", float(info["local_distance"]))
    assert float(info_same["local_distance"]) < 0.004            # only sampling noise
    assert float(info["local_distance"]) == pytest.approx(0.05, abs=0.01)
    assert float(info["local_above"]) > 0.9 and float(info_same["local_above"]) < 0.4
    assert float(apart) < float(same) - 1.0                      # pushed apart: lower loss


def test_local_chamfer_rewards_a_local_feature_over_a_thin_global_change(cfg):
    gen = torch.Generator().manual_seed(0)
    base = sphere(radius=0.45)
    bump, info_bump = L.local_chamfer(_cluster_of([base, _bumped(0.12)]), cfg, gen)
    # a uniform inflation with the same mean surface displacement as the bump
    mean_shift = float(info_bump["local_distance"])
    even, info_even = L.local_chamfer(_cluster_of([base, sphere(radius=0.45 + mean_shift)]), cfg, gen)
    print("bump mean distance", mean_shift, "even shift distance", float(info_even["local_distance"]),
          "loss bump", float(bump), "loss even", float(even))
    assert float(info_even["local_distance"]) == pytest.approx(mean_shift, rel=0.5)
    fb = _cluster_of([base, _bumped(0.12)])
    f = fb.f.detach().clone().requires_grad_(True)
    loss, _ = L.local_chamfer(dataclasses.replace(fb, f=f), cfg, gen)
    (grad,) = torch.autograd.grad(loss, f)
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
    # the gradient is concentrated near the bump (+x side)
    near = (fb.x[0] - torch.tensor([0.45, 0.0, 0.0])).norm(dim=-1) < 0.25
    share = float(grad[:, near].abs().sum() / grad.abs().sum())
    print("share of gradient within 0.25 of the bump:", share, "| share of points there:", float(near.float().mean()))
    assert share > 3 * float(near.float().mean())


def test_local_chamfer_is_safe_without_pairs(cfg):
    fb = _cluster_of([sphere(radius=0.45)])
    loss, info = L.local_chamfer(fb, cfg)
    assert float(loss) == 0.0 and float(info["local_distance"]) == 0.0


# ----------------------------------------------------------- statistic repulsion
def _stretched(scale_y):
    """A smooth ellipsoid-like body squashed or stretched along y."""
    def field(p):
        s = torch.tensor([1.0, scale_y, 1.0]).to(p)
        return 0.45 - torch.sqrt((p * s).square().sum(-1) + 1e-4)
    return field


def _family_fb(scales_y, count=4096, seed=4):
    x = box_points(1, count, seed=seed).expand(len(scales_y), -1, -1).clone()
    codes = unit_codes(len(scales_y), seed=seed)
    parts = [make_fb(_stretched(s), x[i:i + 1], codes[i:i + 1], torch.zeros(1, dtype=torch.long))
             for i, s in enumerate(scales_y)]
    return FieldBatch(codes=codes, cluster=torch.zeros(len(scales_y), dtype=torch.long), x=x,
                      f=torch.cat([p.f for p in parts]), g=torch.cat([p.g for p in parts]),
                      xp=torch.cat([p.xp for p in parts]), fp=torch.cat([p.fp for p in parts]))


def test_statistic_repulsion_rewards_variety_group_by_group(cfg):
    same, _ = L.statistic_repulsion(_family_fb([1.0, 1.0, 1.0, 1.0]), cfg)
    varied, info = L.statistic_repulsion(_family_fb([0.7, 1.0, 1.4, 2.0]), cfg)
    assert float(varied) < float(same) - 1.0          # lower loss for the varied family
    assert set(info) == {"spread_proportions", "spread_thickness", "spread_compactness"}
    assert float(info["spread_proportions"]) > 0.05   # the y-stretch shows up in the proportions


def test_statistic_repulsion_has_diminishing_returns_and_finite_gradients(cfg):
    fb = _family_fb([0.7, 1.0, 1.4, 2.0])
    f = fb.f.detach().clone().requires_grad_(True)
    loss, _ = L.statistic_repulsion(dataclasses.replace(fb, f=f), cfg)
    (grad,) = torch.autograd.grad(loss, f)
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
    flat = _family_fb([1.0, 1.0, 1.0, 1.0])
    f0 = flat.f.detach().clone().requires_grad_(True)
    loss0, _ = L.statistic_repulsion(dataclasses.replace(flat, f=f0), cfg)
    (grad0,) = torch.autograd.grad(loss0, f0)
    assert bool(torch.isfinite(grad0).all())          # identical shapes: finite, thanks to eps
    single = _family_fb([1.0])
    assert float(L.statistic_repulsion(single, cfg)[0]) == 0.0


# ------------------------------------------------------------------------ spread
def _linear_family(rank, members=7, count=2048, seed=5, slope=0.1):
    """One cluster (centre + members). Field = base + slope * modes @ (mix @ code).

    ``rank=32``: 32 orthonormal spatial modes, one per code coordinate, so field differences
    reproduce latent offsets exactly. ``rank=1``: every direction drives the same single mode.
    """
    gen = torch.Generator().manual_seed(seed)
    centre = sample_sphere(1, 32, generator=gen)[0]
    codes = sample_cluster(centre, members + 1, 0.4, generator=gen)
    cluster = torch.zeros(members + 1, dtype=torch.long)
    x = (box_points(1, count, seed=seed) * 0.5).expand(members + 1, -1, -1).clone()   # all inside R_b
    modes = torch.linalg.qr(torch.randn(count, 32, generator=gen))[0] * math.sqrt(count)
    if rank == 1:
        mix = torch.zeros(32, 32)
        mix[0] = torch.randn(32, generator=gen)
    else:
        mix = torch.eye(32)
    f = (slope * (codes @ mix.T) @ modes.T).requires_grad_(True)
    return fb_from_values(f, x, codes, cluster), f


def test_spread_is_zero_for_a_distance_preserving_family(cfg):
    cfg.diversity_pose_invariant = False
    fb, _ = _linear_family(rank=32)
    loss, info = L.spread(fb, cfg)
    assert float(info["pairs"]) == 21                 # 7 members -> 21 pairs
    assert float(loss) < 1e-3


def test_spread_is_large_when_every_direction_makes_the_same_change(cfg):
    cfg.diversity_pose_invariant = False
    fb, f = _linear_family(rank=1)
    loss, info = L.spread(fb, cfg)
    # all patterns are the one mode; the norm floor pulls weakly driven members below 1
    assert float(info["corr_abs_mean"]) > 0.6
    assert float(loss) > 0.3
    (grad,) = torch.autograd.grad(loss, f)
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0


def test_spread_is_safe_with_identical_fields_and_single_codes(cfg):
    cfg.diversity_pose_invariant = False
    fb, f = _linear_family(rank=32)
    same = dataclasses.replace(fb, f=(f.detach()[:1].expand_as(f)).clone().requires_grad_(True))
    loss, _ = L.spread(same, cfg)
    assert bool(torch.isfinite(loss))                 # dead directions read as uncorrelated
    single = dataclasses.replace(fb, cluster=torch.arange(fb.f.shape[0]))
    loss, info = L.spread(single, cfg)
    assert float(loss) == 0.0 and float(info["pairs"]) == 0


# ------------------------------------------------------------ weak-direction pairs
def test_pair_push_value_and_gradient_favour_the_closest_pair(cfg):
    """-log(rate + eps) per pair; the push on a pair grows as its two shapes get more alike."""
    plain = dataclasses.replace(cfg, diversity_pose_invariant=False)
    gen = torch.Generator().manual_seed(0)
    x = (torch.rand(1, 600, 3, generator=gen) * 2 - 1).expand(4, -1, -1).clone()
    radii = torch.tensor([0.50, 0.52, 0.50, 0.60], requires_grad=True)
    fb = make_fb(lambda p: radii[:, None] - torch.sqrt(p.square().sum(-1) + 1e-16), x)
    first, second = torch.tensor([0, 2]), torch.tensor([1, 3])
    loss, info = L.pair_push(fb, first, second, 0.25, plain)
    rates = torch.tensor([0.02, 0.10]) / (plain.diversity_slope * 0.25)
    assert float(loss) == pytest.approx(float(-torch.log(rates + plain.weak_pair_eps).sum()), rel=1e-4)
    assert float(info["pair_rate"]) == pytest.approx(float(rates.mean()), rel=1e-4)
    assert float(info["pair_rate_min"]) == pytest.approx(float(rates.min()), rel=1e-4)
    (grad,) = torch.autograd.grad(loss, radii)
    assert grad[1] < 0 < grad[0] and grad[3] < 0 < grad[2]            # moving a pair apart lowers the loss
    assert abs(float(grad[1])) > 3 * abs(float(grad[3]))              # and the closer pair is pushed harder
    # a chord per pair is accepted too
    again, _ = L.pair_push(fb, first, second, torch.tensor([0.25, 0.25]), plain)
    assert float(again) == pytest.approx(float(loss), rel=1e-6)


def test_pair_push_gives_nothing_for_a_shifted_copy_when_pose_is_removed(cfg):
    gen = torch.Generator().manual_seed(1)
    x = (torch.rand(1, 800, 3, generator=gen) * 2 - 1).expand(2, -1, -1).clone()
    shift = torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    fb = make_fb(lambda p: 0.5 - torch.sqrt((p - shift[:, None]).square().sum(-1) + 1e-16), x)
    first, second = torch.tensor([0]), torch.tensor([1])
    _, kept = L.pair_push(fb, first, second, 0.25, dataclasses.replace(cfg, diversity_pose_invariant=False))
    _, removed = L.pair_push(fb, first, second, 0.25, dataclasses.replace(cfg, diversity_pose_invariant=True))
    assert float(kept["pair_rate"]) > 0.2
    assert float(removed["pair_rate"]) < 0.05 * float(kept["pair_rate"])


def _frame_fb(alpha, beta, seed=2):
    """A centre and two partners: one along a used direction (pattern x), one new (alpha x + beta y)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 1500, 3, generator=gen) * 1.6 - 0.8
    centre = 0.5 - torch.sqrt(x.square().sum(-1) + 1e-16)
    chord_length = 0.25
    used = centre + chord_length * 0.25 * x[..., 0]
    new = centre + chord_length * 0.25 * (alpha * x[..., 0] + beta * x[..., 1])
    f = torch.cat([centre, used, new])
    return fb_from_values(f, x.expand(3, -1, -1), unit_codes(3), torch.zeros(3, dtype=torch.long)), chord_length


def test_frame_volume_gives_nothing_for_a_copied_pattern_and_rewards_a_new_one(cfg):
    plain = dataclasses.replace(cfg, diversity_pose_invariant=False)
    partners = torch.tensor([1, 2])

    def value(alpha, beta):
        fb, chord_length = _frame_fb(torch.tensor(alpha), torch.tensor(beta))
        return L.frame_volume(fb, 0, partners, chord_length, 1, plain)

    none, _ = value(0.0, 0.0)
    copy, copy_info = value(0.5, 0.0)                      # the new partner replays the used pattern
    fresh, fresh_info = value(0.0, 0.5)                    # the new partner does something else
    assert float(none) - float(copy) < 0.15                # a copy earns next to nothing
    assert float(copy) - float(fresh) > 0.5                # a new pattern earns a lot
    assert float(copy_info["new_raw_rms"]) == pytest.approx(float(fresh_info["new_raw_rms"]), rel=0.1)
    assert float(copy_info["new_rms"]) < 0.3 * float(fresh_info["new_rms"])   # what counts is what is left
    assert float(fresh_info["used_rms_min"]) > 0.05


def test_frame_volume_pushes_the_new_part_harder_than_the_copied_part(cfg):
    plain = dataclasses.replace(cfg, diversity_pose_invariant=False)
    alpha = torch.tensor(0.5, requires_grad=True)
    beta = torch.tensor(0.05, requires_grad=True)          # ten times smaller than the copied part
    fb, chord_length = _frame_fb(alpha, beta)
    loss, _ = L.frame_volume(fb, 0, torch.tensor([1, 2]), chord_length, 1, plain)
    d_alpha, d_beta = torch.autograd.grad(loss, [alpha, beta])
    assert d_beta < 0                                      # growing the new pattern lowers the loss
    assert abs(float(d_beta)) > abs(float(d_alpha))


# ------------------------------------------------------------------ anchor objects
def _linear_grids(resolution=9):
    """Two fields that are linear in space, stored the way the anchor file stores them."""
    axis = torch.linspace(-1.0, 1.0, resolution)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([0.3 * gx - 0.2 * gy + 0.1 * gz + 0.05, -0.1 * gx + 0.4 * gy + 0.25 * gz - 0.02])


def test_sample_grid_reads_x_y_z_in_the_stored_order():
    grids = _linear_grids()
    gen = torch.Generator().manual_seed(4)
    pts = torch.rand(2, 50, 3, generator=gen) * 2 - 1
    expected = torch.stack([0.3 * pts[0, :, 0] - 0.2 * pts[0, :, 1] + 0.1 * pts[0, :, 2] + 0.05,
                            -0.1 * pts[1, :, 0] + 0.4 * pts[1, :, 1] + 0.25 * pts[1, :, 2] - 0.02])
    assert torch.allclose(L.sample_grid(grids, pts), expected, atol=1e-5)     # trilinear is exact on linear fields
    outside = torch.tensor([[[3.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]])
    edge = torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    assert torch.allclose(L.sample_grid(grids, outside), L.sample_grid(grids, edge))   # clamped to the box
    moving = pts.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(L.sample_grid(grids, moving)[0].sum(), moving)
    assert torch.allclose(grad[0], torch.tensor([0.3, -0.2, 0.1]).expand(50, -1), atol=1e-4)


def test_anchor_match_is_zero_on_the_target_and_pulls_the_field_towards_it(cfg):
    grids = torch.stack([0.5 - torch.sqrt(sum(g.square() for g in torch.meshgrid(
        *([torch.linspace(-1.0, 1.0, 33)] * 3), indexing="ij")) + 1e-12)] * 2)        # two balls of radius 0.5
    gen = torch.Generator().manual_seed(6)
    x = (torch.rand(1, 700, 3, generator=gen) * 2 - 1).expand(3, -1, -1).clone()
    radii = torch.tensor([0.9, 0.5, 0.6], requires_grad=True)       # row 0 is not an anchor
    fb = make_fb(lambda p: radii[:, None] - torch.sqrt(p.square().sum(-1) + 1e-16), x)
    rows = torch.tensor([1, 2])
    loss, info = L.anchor_match(fb, rows, grids, cfg)
    exact, exact_info = L.anchor_match(fb, torch.tensor([1, 1]), grids, cfg)
    assert float(exact) < 0.01 and float(exact_info["iou"]) > 0.97   # only the grid's interpolation error is left
    assert float(loss) > float(exact) + 0.03
    assert 0.3 < float(info["iou"]) < 0.9
    (grad,) = torch.autograd.grad(loss, radii)
    assert grad[2] > 0                                              # the ball that is too large is told to shrink
    assert grad[0] == 0                                             # rows that are not anchors are untouched
    # per-row weights scale the pull; the reported parts lean on the heavier rows
    half, half_info = L.anchor_match(fb, rows, grids, cfg, weights=torch.tensor([1.0, 0.5]))
    gone, gone_info = L.anchor_match(fb, rows, grids, cfg, weights=torch.tensor([1.0, 0.0]))
    assert float(gone) < float(half) < float(loss)
    assert float(gone_info["distance"]) < float(half_info["distance"]) < float(info["distance"])


# ------------------------------------------------------------- sparse curvature
class _Analytic:
    """A stand-in decoder whose field is a given function of position (the code is ignored)."""

    def __init__(self, fn):
        self.fn = fn

    def prepare(self, codes):
        from arm_planner.data_free.decoder import Modulation
        return Modulation(codes.new_zeros(codes.shape[0], 1, 1), codes.new_zeros(codes.shape[0], 1, 1))

    def query(self, mod, x):
        return self.fn(x)


def _crease_of(fn, cfg, seed=0, rows=None):
    gen = torch.Generator().manual_seed(seed)
    codes = sample_sphere(4, cfg.latent_dim, generator=gen)
    x = sample_points(cfg, 4, gen, "cpu")
    decoder = _Analytic(fn)
    fb = evaluate_field(decoder, codes, torch.arange(4), x, cfg, create_graph=True)
    return L.crease(decoder, fb, cfg, generator=gen, rows=rows)


def test_crease_term_is_zero_on_a_plane_follows_the_radius_on_a_ball_and_is_low_on_a_cube():
    cfg = DataFreeConfig(points=4096, crease_points=512, crease_step=0.02, crease_scale=3.0, crease_band=0.04)
    plane, info = _crease_of(lambda x: 0.1 - x[..., 2], cfg)
    assert float(plane) < 1e-6 and float(info["flat"]) == pytest.approx(1.0) and float(info["used"]) > 0.5
    values = {}
    for radius in (0.25, 0.5):
        ball, info = _crease_of(lambda x, r=radius: r - x.norm(dim=-1), cfg)
        bend = 1.0 / radius                                    # the normal of a ball turns by 1 / radius per length
        assert float(ball) == pytest.approx(bend ** 2 / (bend ** 2 + 9.0), rel=0.02)
        assert float(info["turn"]) == pytest.approx(bend, rel=0.02) and float(info["used"]) > 0.9
        values[radius] = float(ball)
    assert values[0.25] > values[0.5]
    # A cube of about the volume of the 0.4 ball: only pairs across an edge pay, and they pay the full 1.
    cube, info = _crease_of(lambda x: 0.32 - x.abs().amax(dim=-1), cfg)
    ball, _ = _crease_of(lambda x: 0.4 - x.norm(dim=-1), cfg)
    assert 0.01 < float(cube) < 0.12 and float(ball) > 4 * float(cube)
    assert float(info["sharp"]) == pytest.approx(float(cube), abs=0.01) and float(info["flat"]) > 0.85


def test_crease_term_prefers_a_sharper_edge_at_the_same_size():
    # Rounded boxes |x|_p = h from p = 2 (ball) upward: the higher p, the flatter the faces and the
    # tighter the edges. The term must fall at every step of the way, with no hump in between.
    cfg = DataFreeConfig(points=4096, crease_points=512)
    seen = []
    for power in (2.0, 3.0, 5.0, 10.0, 30.0):
        value, _ = _crease_of(lambda x, p=power: 0.4 - (x.abs().clamp_min(1e-6) ** p).sum(dim=-1) ** (1.0 / p), cfg)
        seen.append(float(value))
    assert seen == sorted(seen, reverse=True) and seen[0] > 1.8 * seen[-1]


def test_crease_term_leaves_out_rows_and_reaches_the_real_decoder():
    cfg = dataclasses.replace(_small_cfg(), crease_points=64)
    decoder, codes, cluster, x = _real_batch(cfg, "cpu")
    fb = evaluate_field(decoder, codes, cluster, x, cfg, create_graph=True)
    value, info = L.crease(decoder, fb, cfg, torch.Generator().manual_seed(2))
    # the untrained decoder is the ball of radius base_radius: bend 2, cost 4 / 13
    assert float(value) == pytest.approx(4.0 / 13.0, rel=0.1)
    value.backward()
    assert sum(float(p.grad.abs().sum()) for p in decoder.parameters() if p.grad is not None) > 0
    none, info = L.crease(decoder, fb, cfg, rows=0)
    assert float(none) == 0.0 and float(info["used"]) == 0.0
    some, _ = L.crease(decoder, fb, cfg, torch.Generator().manual_seed(2), rows=4)
    assert torch.isfinite(some) and float(some) > 0
