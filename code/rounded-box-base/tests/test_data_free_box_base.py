"""Tests for the rounded-box base of the data-free decoder (``base_shape="rounded_box"``)."""

import math
import os

import pytest
import torch

from arm_planner.data_free import train as T
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.decoder import (
    BASE_TAU,
    ModulatedSphereDecoder,
    core_point,
    rounded_box_field,
)
from arm_planner.data_free.latent import sample_sphere

DECODER = dict(width=64, layers=8, skip_layer=4, mapping_width=32, points=256)
TRAIN = dict(
    width=32, layers=3, mapping_width=32, clusters=2, codes_per_cluster=3, points=256,
    bank_size=16, hard_negatives=8, grad_change_points=32, steps=20, lr_warmup_steps=5,
    diag_every=0, ckpt_every=0, probe_codes=4, diag_resolution=16,
    dead_direction_probes=1, dead_direction_points=512,
)


def box_cfg(**kw):
    return DataFreeConfig(**{**DECODER, "base_shape": "rounded_box", **kw})


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def codes(n, dim=32, seed=0):
    return sample_sphere(n, dim, gen(seed), torch.device("cpu"))


def scramble_base_head(dec, scale=3.0, seed=1):
    """Large random base-head weights: the parameters go well away from the sphere."""
    with torch.no_grad():
        g = gen(seed)
        dec.base_head.weight.copy_(torch.randn(dec.base_head.weight.shape, generator=g) * scale)
        dec.base_head.bias.copy_(torch.randn(dec.base_head.bias.shape, generator=g) * scale)


# ------------------------------------------------------------------ the box field
def test_rounded_box_with_no_extent_is_the_sphere_field():
    x = torch.rand(2, 500, 3, generator=gen()) * 2 - 1
    base = torch.tensor([[0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.3]])
    want = base[:, 3:] - torch.sqrt((x * x).sum(-1) + 1e-8)
    assert torch.equal(rounded_box_field(x, base), want)


def test_rounded_box_field_is_the_exact_distance():
    b1, b2, b3, r = 0.3, 0.2, 0.1, 0.05
    base = torch.tensor([[b1, b2, b3, r]])
    pts = torch.tensor([[
        [b1 + r + 0.2, 0.0, 0.0],                 # beyond the x face: distance 0.2
        [0.0, 0.0, 0.0],                           # centre: nearest face is z, depth b3 + r
        [b1 + 0.1, b2 + 0.2, b3 + 0.2],           # corner region: r - |(0.1, 0.2, 0.2)|
        [0.1, 0.05, b3 + r],                       # on the z face
    ]])
    f = rounded_box_field(pts, base)[0]
    # within 1e-4 (the sqrt(. + 1e-8) that keeps b = 0 identical to the sphere field)
    assert f[0] == pytest.approx(-0.2, abs=2e-4)
    assert f[1] == pytest.approx(b3 + r, abs=2e-4)
    assert f[2] == pytest.approx(r - 0.3, abs=2e-4)
    assert f[3] == pytest.approx(0.0, abs=2e-4)


def test_rounded_box_field_has_unit_slope():
    base = torch.tensor([[0.4, 0.15, 0.0, 0.08]])
    x = (torch.rand(1, 4000, 3, generator=gen()) * 2 - 1).requires_grad_(True)
    (g,) = torch.autograd.grad(rounded_box_field(x, base).sum(), x)
    norm = g.norm(dim=-1)[0]
    # 1 everywhere except on the core's medial set (a set of measure zero)
    assert (norm - 1).abs().max() < 1e-4


def test_core_point_is_the_nearest_point_of_the_core():
    base = torch.tensor([[0.3, 0.1, -0.005, 0.1]])
    x = torch.tensor([[[0.5, -0.5, 0.2], [0.1, 0.05, 0.0]]])
    assert torch.allclose(core_point(x, base), torch.tensor([[[0.3, -0.1, 0.0], [0.1, 0.05, 0.0]]]))


# ------------------------------------------------------------------ parameters
def test_new_box_decoder_starts_at_the_sphere():
    dec = ModulatedSphereDecoder(box_cfg())
    mod = dec.prepare(codes(16))
    assert torch.equal(mod.base[:, :3], torch.zeros(16, 3))
    assert torch.allclose(mod.base[:, 3], torch.full((16,), 0.5), atol=1e-6)


def test_base_parameters_are_ordered_and_within_their_limits():
    cfg = box_cfg(base_round_min=0.05)
    dec = ModulatedSphereDecoder(cfg)
    scramble_base_head(dec)
    base = dec.prepare(codes(512)).base
    b, r = base[:, :3], base[:, 3]
    slack = 2 * BASE_TAU * math.log(2.0) + 1e-6
    assert (b[:, 0] >= b[:, 1] - slack).all() and (b[:, 1] >= b[:, 2] - slack).all()
    assert (b >= -BASE_TAU * math.log(2.0) - 1e-6).all()
    assert (r >= cfg.base_round_min - 1e-6).all()
    assert (b.clamp_min(0).norm(dim=-1) + r <= cfg.max_reach() + 1e-5).all()


def test_head_output_reaches_rods_sheets_and_the_reach_limit():
    cfg = box_cfg(base_round_min=0.05)
    dec = ModulatedSphereDecoder(cfg)
    with torch.no_grad():
        dec.base_head.bias.copy_(torch.tensor([5.0, -5.0, -5.0, -20.0]))      # long, thin: a rod
    rod = dec.prepare(codes(1)).base[0].detach()
    assert float(rod[3]) == pytest.approx(cfg.base_round_min, abs=1e-4)
    assert float(rod[0] + rod[3]) == pytest.approx(cfg.max_reach(), abs=1e-4)
    assert abs(float(rod[1])) < 0.01 and abs(float(rod[2])) < 0.01
    with torch.no_grad():
        dec.base_head.bias.copy_(torch.tensor([-5.0, 5.0, -5.0, -20.0]))      # two long axes: a sheet
    sheet = dec.prepare(codes(1)).base[0].detach()
    assert float(sheet[0]) == pytest.approx(float(sheet[1]), abs=0.01)
    assert float(sheet[:3].clamp_min(0).norm() + sheet[3]) == pytest.approx(cfg.max_reach(), abs=1e-4)
    assert abs(float(sheet[2])) < 0.01


def test_base_range_zero_gives_the_sphere_whatever_the_head():
    dec = ModulatedSphereDecoder(box_cfg(base_ramp_steps=100))
    assert float(dec.base_range) == 0.0
    scramble_base_head(dec)
    base = dec.prepare(codes(32)).base
    assert torch.equal(base[:, :3], torch.zeros(32, 3))
    assert torch.allclose(base[:, 3], torch.full((32,), 0.5))
    dec.base_range.fill_(0.5)
    half = dec.prepare(codes(32)).base
    dec.base_range.fill_(1.0)
    full = dec.prepare(codes(32)).base
    assert torch.allclose(half[:, :3], 0.5 * full[:, :3], atol=1e-6)
    assert torch.allclose(half[:, 3] - 0.5, 0.5 * (full[:, 3] - 0.5), atol=1e-6)


def test_base_speed_scales_the_head_output():
    a = ModulatedSphereDecoder(box_cfg())
    b = ModulatedSphereDecoder(box_cfg(base_speed=2.0))
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        a.base_head.bias.fill_(0.01)
        b.base_head.bias.fill_(0.005)
    assert torch.allclose(a.prepare(codes(4)).base, b.prepare(codes(4)).base)


# ------------------------------------------------------------------ field guarantees
@pytest.mark.parametrize("bias", [[0.0, 0.0, 0.4, -3.0],      # cube with sharp corners
                                  [0.6, -1.0, -1.0, -2.0]])   # rod
def test_shape_stays_inside_the_base_grown_by_the_correction_bound(bias):
    cfg = box_cfg(output_init_std=1.0)
    dec = ModulatedSphereDecoder(cfg)
    scramble_base_head(dec, scale=0.1)
    with torch.no_grad():
        dec.base_head.bias.copy_(torch.tensor(bias))
    c = codes(8)
    mod = dec.prepare(c)
    x = torch.rand(8, 4000, 3, generator=gen(3)) * 2 - 1
    with torch.no_grad():
        f = dec.query(mod, x)
        outside = rounded_box_field(x, mod.base) < -cfg.correction_bound
        beyond = x.norm(dim=-1) > cfg.bound_radius
        corner = mod.base[:, :3].clamp_min(0).norm(dim=-1) + mod.base[:, 3]
    assert float(corner.max()) > 0.5
    assert outside.any() and (f[outside] < 0).all()
    assert (f[beyond] < 0).all()


def test_inner_fraction_keeps_the_core_inside():
    k = 0.6
    dec = ModulatedSphereDecoder(box_cfg(output_init_std=1.0, base_inner_fraction=k))
    scramble_base_head(dec, scale=0.1)
    with torch.no_grad():
        dec.base_head.bias.copy_(torch.tensor([0.3, 0.2, 0.0, -4.0]))
    assert float(dec.prepare(codes(8)).base[:, 0].min()) > 0.2
    with torch.no_grad():
        mod = dec.prepare(codes(8))
        core = (torch.rand(8, 2000, 3, generator=gen(4)) * 2 - 1) * mod.base[:, None, :3].clamp_min(0)
        f = dec.query(mod, core)
    # less up to tau ln 2 (an extent may sit that far below zero) and 1e-4 (the sqrt offset)
    assert (f >= (1 - k) * mod.base[:, None, 3] - BASE_TAU * math.log(2.0) - 2e-4).all()


def test_gradient_reaches_the_base_head_and_the_feature_columns():
    dec = ModulatedSphereDecoder(box_cfg())
    x = torch.rand(4, 300, 3, generator=gen()) * 2 - 1
    (dec(codes(4), x) ** 2).mean().backward()
    assert dec.base_head.weight.grad.abs().sum() > 0
    assert dec.hidden[0].weight.grad[:, -4:].abs().sum() > 0
    assert dec.hidden[4].weight.grad[:, -4:].abs().sum() > 0


def test_sphere_decoder_is_unchanged_by_default():
    dec = ModulatedSphereDecoder(DataFreeConfig(**DECODER))
    assert not hasattr(dec, "base_head")
    assert dec.prepare(codes(2)).base is None
    assert "base_range" not in dec.state_dict()


# ------------------------------------------------------------------ config
def test_config_checks_the_base_settings():
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="cube")
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="rounded_box", base_max_reach=0.7)       # 0.7 + 0.35 > 0.9
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="rounded_box", base_round_min=0.6)
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="rounded_box", use_window=True)
    assert DataFreeConfig().max_reach() == pytest.approx(0.55)
    assert DataFreeConfig(correction_bound=0.2).max_reach() == pytest.approx(0.7)
    # an override of correction_bound moves the default reach with it
    cfg = T.apply_overrides(DataFreeConfig(base_shape="rounded_box"), ["correction_bound=0.2"])
    assert cfg.max_reach() == pytest.approx(0.7)


# ------------------------------------------------------------------ trainer
def test_warm_start_from_a_sphere_checkpoint_reproduces_its_field(tmp_path):
    sphere = T.Trainer(DataFreeConfig(**{**TRAIN, "layers": 6, "skip_layer": 3}))
    for _ in range(2):
        sphere.step()
    path = str(tmp_path / "sphere.pt")
    sphere.save_checkpoint(path)
    box = T.Trainer(DataFreeConfig(**{**TRAIN, "layers": 6, "skip_layer": 3, "base_shape": "rounded_box",
                                      "init_model_from": path}))
    c, x = codes(6), torch.rand(6, 400, 3, generator=gen()) * 2 - 1
    with torch.no_grad():
        assert torch.allclose(box.model(c, x), sphere.model(c, x), atol=1e-6)
    assert box.multipliers.state_dict()["multipliers"] == sphere.multipliers.state_dict()["multipliers"]
    assert box.step_count == 0
    box.step()                                       # and it trains on from there


def test_warm_start_rejects_a_different_network(tmp_path):
    other = T.Trainer(DataFreeConfig(**{**TRAIN, "width": 16}))
    path = str(tmp_path / "other.pt")
    other.save_checkpoint(path)
    with pytest.raises(ValueError):
        T.Trainer(DataFreeConfig(**{**TRAIN, "base_shape": "rounded_box", "init_model_from": path}))


def test_box_trainer_steps_ramps_and_resumes(tmp_path):
    cfg = DataFreeConfig(**{**TRAIN, "base_shape": "rounded_box", "base_ramp_steps": 4})
    trainer = T.Trainer(cfg, output_dir=str(tmp_path), tensorboard=False)
    ranges = []
    for _ in range(3):
        record = trainer.step()
        assert math.isfinite(record["loss"])
        ranges.append(float(trainer.model.base_range))
    assert ranges == [0.0, 0.25, 0.5]
    trainer.save_checkpoint()
    again = T.Trainer.from_checkpoint(str(tmp_path / T.CHECKPOINT_NAME))
    assert float(again.model.base_range) == 0.5
    again.step()
    assert float(again.model.base_range) == 0.75


def test_base_surface_rows_lie_near_base_surfaces():
    cfg = DataFreeConfig(**{**TRAIN, "base_shape": "rounded_box", "base_surface_points": 40,
                            "base_surface_noise": 0.0})
    trainer = T.Trainer(cfg)
    scramble_base_head(trainer.model, scale=1.0)
    batch = trainer.sample_batch()
    n, h = cfg.points, cfg.hard_negatives
    rows = batch.x[:, n - h - 40: n - h]
    with torch.no_grad():
        base = trainer.model.prepare(batch.codes).base
    first = torch.tensor([int((batch.cluster == c).nonzero()[0]) for c in range(batch.x.shape[0])])
    f = rounded_box_field(rows, base[first])
    inside_box = (rows.abs() < 1.0 - 1e-6).all(dim=-1)
    assert (f[inside_box].abs() < 1e-4).all()
    assert torch.equal(batch.x[:, 0], torch.zeros_like(batch.x[:, 0]))       # origin row untouched


def test_base_surface_rows_with_shared_points():
    cfg = DataFreeConfig(**{**TRAIN, "base_shape": "rounded_box", "base_surface_points": 32,
                            "global_codes": True})
    trainer = T.Trainer(cfg)
    batch = trainer.sample_batch()
    assert batch.x.shape[0] == 1
    assert torch.isfinite(batch.x).all()
    trainer.step()


def test_diagnostics_report_the_base():
    cfg = DataFreeConfig(**{**TRAIN, "base_shape": "rounded_box"})
    trainer = T.Trainer(cfg)
    scramble_base_head(trainer.model, scale=0.5)
    out = trainer.run_diagnostics()
    for key in ("base/r_mean", "base/b1_max", "base/elongation_mean", "base/rod_fraction",
                "base/sheet_fraction", "base/field_variance_share", "base/range"):
        assert key in out and math.isfinite(out[key]), key
    assert 0.0 <= out["base/field_variance_share"]
    assert "base r" in T.diagnostics_line(out)


def test_cli_starts_from_another_runs_config(tmp_path):
    first = tmp_path / "first"
    T.main(["--output-dir", str(first), "--steps", "1", "--no-tensorboard", "--quiet", "--device", "cpu"]
           + [f"--set={k}={v}" for k, v in {**TRAIN, "width": 24}.items()])
    second = tmp_path / "second"
    trainer = T.main(["--output-dir", str(second), "--steps", "1", "--no-tensorboard", "--quiet", "--device", "cpu",
                      "--config-from", str(first / "config.json"), "--set", "base_shape=rounded_box",
                      "--set", f"init_model_from={first / T.CHECKPOINT_NAME}"])
    assert trainer.cfg.width == 24 and trainer.cfg.base_shape == "rounded_box"
    assert trainer.step_count == 1


# ------------------------------------------------------------------ fixed code-to-base map
def fixed_cfg(**kw):
    return box_cfg(base_map="fixed", **kw)


def test_fixed_map_has_no_weights_and_spreads_the_bases():
    cfg = fixed_cfg()
    dec = ModulatedSphereDecoder(cfg)
    assert not hasattr(dec, "base_head") and "base_projection" in dec.state_dict()
    P = dec.base_projection
    assert torch.allclose(P.T @ P, torch.eye(4), atol=1e-5)
    with torch.no_grad():
        base = dec.prepare(codes(4000)).base
    b, r = base[:, :3], base[:, 3]
    assert (b >= 0).all() and (r >= cfg.base_round_min - 1e-6).all() and (r <= cfg.base_radius + 1e-6).all()
    assert (b.norm(dim=-1) + r <= cfg.max_reach() + 1e-5).all()
    long = (b > 2 * r[:, None]).sum(-1)
    for k in range(3):                                   # blobs, rods and sheets all occur in quantity
        assert (long == k).float().mean() > 0.1
    # a rod lies along any axis: no axis order
    rods = long == 1
    assert len(set(b[rods].argmax(-1).tolist())) == 3


def test_fixed_map_is_set_by_the_seed_and_ramps_from_the_sphere():
    a = ModulatedSphereDecoder(fixed_cfg())
    b = ModulatedSphereDecoder(fixed_cfg())
    c = ModulatedSphereDecoder(fixed_cfg(base_map_seed=1))
    z = codes(8)
    with torch.no_grad():
        assert torch.equal(a.prepare(z).base, b.prepare(z).base)
        assert not torch.allclose(a.prepare(z).base, c.prepare(z).base)
    ramped = ModulatedSphereDecoder(fixed_cfg(base_ramp_steps=10))
    with torch.no_grad():
        base = ramped.prepare(z).base
    assert torch.equal(base[:, :3], torch.zeros(8, 3)) and torch.allclose(base[:, 3], torch.full((8,), 0.5))


def test_scores_for_base_inverts_the_fixed_map():
    dec = ModulatedSphereDecoder(fixed_cfg())
    with torch.no_grad():
        base = dec.prepare(codes(200, seed=5)).base
        s = dec.scores_for_base(base)
        D = dec.latent_dim
        z = torch.zeros(200, D)
        z += s @ dec.base_projection.T / math.sqrt(D)          # base components exactly s
        rest = codes(200, seed=6)
        rest = rest - (rest @ dec.base_projection) @ dec.base_projection.T
        scale = torch.sqrt(1 - (s / math.sqrt(D)).square().sum(-1)) / rest.norm(dim=-1)
        again = dec.prepare(z + rest * scale[:, None]).base
    a = base[:, :3] / (dec.base_max_reach - base[:, 3:])
    frac = (base[:, 3] - dec.base_round_min) / (dec.map_round_max - dec.base_round_min)
    keep = (s.abs() < 2.49).all(-1) & (a < 0.949).all(-1) & (frac > 0.021) & (frac < 0.979)  # within the limits
    assert keep.float().mean() > 0.5
    assert torch.allclose(again[keep], base[keep], atol=1e-3)     # extents under 0.1% of the room are floored


def test_fit_rounded_boxes_recovers_a_known_box():
    true = torch.tensor([[0.3, 0.1, 0.0, 0.08], [0.0, 0.0, 0.0, 0.4], [0.05, 0.25, 0.25, 0.06]])
    axis = torch.linspace(-1, 1, 49)
    X = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)
    grids = rounded_box_field(X.expand(3, -1, -1), true).reshape(3, 49, 49, 49)
    fit, err = T.fit_rounded_boxes(grids, gen())
    assert (err < 0.005).all()
    # a sphere and a slightly rounded small cube have nearly the same surface, so the parameters
    # are only loosely set; the fitted surface is what counts
    assert torch.allclose(fit, true, atol=0.03)
    near = grids.reshape(3, -1).abs() < 0.1
    fitted = rounded_box_field(X.expand(3, -1, -1), fit)
    assert ((fitted - grids.reshape(3, -1)).abs()[near] < 0.02).all()


def test_anchor_codes_are_given_bases_that_fit_their_objects(tmp_path):
    axis = torch.linspace(-1, 1, 33)
    X = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)
    true = torch.tensor([[0.35, 0.0, 0.0, 0.1], [0.0, 0.3, 0.3, 0.07], [0.0, 0.0, 0.0, 0.35]])
    grids = rounded_box_field(X.expand(3, -1, -1), true).reshape(3, 33, 33, 33)
    path = str(tmp_path / "anchors.pt")
    torch.save({"grid": grids.half(), "resolution": 33, "positive_inside": True, "file_ids": [0, 1, 2]}, path)
    cfg = DataFreeConfig(**{**TRAIN, "latent_dim": 16, "global_codes": True, "diversity_mode": "nearest",
                            "batch_neighbours": True, "accumulate": 2, "anchor_path": path,
                            "anchor_codes_per_pass": 1, "base_shape": "rounded_box", "base_map": "fixed",
                            "anchor_base_fit": True})
    trainer = T.Trainer(cfg, output_dir=str(tmp_path / "run"), tensorboard=False)
    with torch.no_grad():
        got = trainer.model.prepare(trainer.anchor_codes).base
    assert torch.allclose(trainer.anchor_codes.norm(dim=-1), torch.ones(3), atol=1e-5)
    target = torch.tensor(trainer.anchor_base_fit["target"])
    assert torch.allclose(got, target, atol=1e-3)                  # each code gives back its fitted box
    assert torch.allclose(target, true, atol=0.03)
    assert os.path.exists(tmp_path / "run" / "anchor_base_fit.json")
    trainer.step()


def test_fixed_map_warm_starts_from_a_sphere_checkpoint(tmp_path):
    sphere = T.Trainer(DataFreeConfig(**TRAIN))
    path = str(tmp_path / "sphere.pt")
    sphere.save_checkpoint(path)
    box = T.Trainer(DataFreeConfig(**{**TRAIN, "base_shape": "rounded_box", "base_map": "fixed",
                                      "base_ramp_steps": 5, "init_model_from": path}))
    c, x = codes(6), torch.rand(6, 300, 3, generator=gen()) * 2 - 1
    with torch.no_grad():
        assert torch.allclose(box.model(c, x), sphere.model(c, x), atol=1e-6)    # ramp 0: the sphere
    for _ in range(3):
        assert math.isfinite(box.step()["loss"])


def test_config_checks_the_fixed_map():
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="rounded_box", base_map="random")
    with pytest.raises(ValueError):
        DataFreeConfig(anchor_base_fit=True)                                     # needs the fixed map
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="rounded_box", base_map="fixed", base_map_round_max=0.04)


# ------------------------------------------------------------------ linear head with a deviation budget
def linear_cfg(**kw):
    return box_cfg(base_head_form="linear", **kw)


def test_linear_head_starts_exactly_at_the_sphere():
    dec = ModulatedSphereDecoder(linear_cfg())
    with torch.no_grad():
        base = dec.prepare(codes(8)).base
    assert torch.allclose(base[:, :3], torch.zeros(8, 3), atol=1e-6)
    assert torch.allclose(base[:, 3], torch.full((8,), 0.5), atol=1e-6)


def test_linear_head_passes_gradient_where_b1_went_flat():
    dec = ModulatedSphereDecoder(linear_cfg())
    # B1's head ended at u = -0.065 on every extent (slope there 0.0015); the linear form keeps it near 1
    # (rows with room for the extents: r well below the reach, so the corner cap is not active)
    u = torch.tensor([[-0.065, 0.0, 0.1, -0.2], [0.1, -0.05, 0.0, -0.3]], requires_grad=True)
    base = dec._linear_base(u)
    (gb,) = torch.autograd.grad(base[0, 0], u, retain_graph=True)
    assert gb[0, 0] > 0.75
    (gr,) = torch.autograd.grad(base[1, 3], u)
    assert gr[1, 3] > 0.9                                         # r = 0.2: still about slope 1
    assert torch.allclose(base[1, 3], torch.tensor(0.2), atol=5e-3)          # the floor's knee bends it by 0.0025
    assert torch.allclose(base[0, :3], torch.tensor([-0.065, 0.0, 0.1]), atol=5e-3)  # lens floor's knee: -0.062
    # at the sphere the radius moves with slope about 1 in both directions
    u0 = torch.zeros(1, 4, requires_grad=True)
    (g0,) = torch.autograd.grad(dec._linear_base(u0)[0, 3], u0)
    assert g0[0, 3] > 0.98


def test_linear_head_keeps_its_hard_limits():
    cfg = linear_cfg()
    dec = ModulatedSphereDecoder(cfg)
    u = torch.tensor([[2.0, 2.0, 2.0, -0.2], [-5.0, -5.0, -5.0, -5.0], [0.3, 0.0, 0.0, 3.0]])
    base = dec._linear_base(u)
    assert (base[:, :3].clamp_min(0).norm(dim=-1) + base[:, 3] <= cfg.max_reach() + 1e-4).all()
    assert (base[:, 3] >= cfg.base_round_min - 1e-4).all() and (base[:, 3] <= cfg.max_reach() + 1e-4).all()
    assert (base[:, :3] > -0.11).all()


def test_budget_holds_the_base_then_lets_it_go(tmp_path):
    cfg = DataFreeConfig(**{**TRAIN, "base_shape": "rounded_box", "base_head_form": "linear",
                            "base_budget": 0.2, "base_budget_steps": 4, "base_budget_start": 0.01})
    trainer = T.Trainer(cfg, output_dir=str(tmp_path), tensorboard=False)
    assert trainer.multipliers.multiplier("base_deviation") == pytest.approx(1000.0)
    tolerances, records = [], []
    for _ in range(5):
        records.append(trainer.step())
        tolerances.append(records[-1]["tol/base_deviation"])
    assert tolerances[0] == pytest.approx(0.01 ** 2)
    assert tolerances[-1] == pytest.approx(0.2 ** 2)
    for record in records:
        assert math.isfinite(record["base/deviation"]) and math.isfinite(record["mult/base_deviation"])
    # under budget: the weight decays
    assert records[-1]["mult/base_deviation"] < 1000.0
    trainer.save_checkpoint()
    again = T.Trainer.from_checkpoint(str(tmp_path / T.CHECKPOINT_NAME))
    assert again.multipliers.multiplier("base_deviation") == pytest.approx(trainer.multipliers.multiplier("base_deviation"))


def test_budget_survives_a_warm_start_with_accumulation(tmp_path):
    wide = dict(global_codes=True, diversity_mode="nearest", accumulate=2, batch_neighbours=True,
                pool_points=40, nearest_k=3)
    sphere = T.Trainer(DataFreeConfig(**{**TRAIN, **wide}))
    sphere.step()
    path = str(tmp_path / "s.pt")
    sphere.save_checkpoint(path)
    box = T.Trainer(DataFreeConfig(**{**TRAIN, **wide, "base_shape": "rounded_box", "base_head_form": "linear",
                                      "base_budget": 0.2, "init_model_from": path}))
    assert box.multipliers.multiplier("base_deviation") == pytest.approx(1000.0)
    record = box.step()
    assert math.isfinite(record["loss"]) and "base/deviation" in record and "mult/base_deviation" in record
    c, x = codes(4), torch.rand(4, 200, 3, generator=gen()) * 2 - 1


def test_config_checks_the_budget():
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="rounded_box", base_head_form="sideways")
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="rounded_box", base_budget=0.2)            # needs the linear head
    DataFreeConfig(base_shape="rounded_box", base_head_form="linear", base_budget=0.2)


# ------------------------------------------------------------------ separate base network, raw parameter inputs
def test_separate_base_network_starts_at_the_sphere_and_shares_nothing():
    cfg = linear_cfg(base_head_net="separate")
    dec = ModulatedSphereDecoder(cfg)
    assert not hasattr(dec, "base_head") and isinstance(dec.base_net[1], torch.nn.Tanh)
    z = codes(6)
    mod = dec.prepare(z)
    assert torch.allclose(mod.base[:, :3], torch.zeros(6, 3), atol=1e-6)
    assert torch.allclose(mod.base[:, 3], torch.full((6,), 0.5), atol=1e-6)
    with torch.no_grad():
        dec.base_net[-1].weight.normal_(0, 0.5)                 # move off the start
    mod = dec.prepare(z)
    assert float(mod.base.std(dim=0).mean()) > 1e-3             # bases differ between codes
    mod.base.sum().backward()
    assert dec.base_net[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in dec.mapping.parameters())     # the side network is untouched
    assert all(p.grad is None for p in dec.film.parameters())


def test_raw_parameter_inputs_widen_the_input_and_start_at_zero(tmp_path):
    plain = ModulatedSphereDecoder(linear_cfg())
    raw = ModulatedSphereDecoder(linear_cfg(base_param_inputs=True))
    assert raw.input_dim == plain.input_dim + 4
    assert (raw.hidden[0].weight[:, -8:] == 0).all() and (raw.hidden[4].weight[:, -8:] == 0).all()
    x = torch.rand(3, 100, 3, generator=gen()) * 2 - 1
    raw(codes(3), x).square().mean().backward()
    # at the sphere the parameter inputs (deviations) are all zero, so their weights get no gradient yet
    assert raw.hidden[0].weight.grad[:, -4:].abs().sum() == 0
    raw.zero_grad()
    with torch.no_grad():
        raw.base_head.bias.copy_(torch.tensor([0.02, -0.01, 0.0, -0.1]))
    raw(codes(3), x).square().mean().backward()
    assert raw.hidden[0].weight.grad[:, -4:].abs().sum() > 0     # once the base moves, they learn
    sphere = T.Trainer(DataFreeConfig(**TRAIN))
    path = str(tmp_path / "s.pt")
    sphere.save_checkpoint(path)
    box = T.Trainer(DataFreeConfig(**{**TRAIN, "base_shape": "rounded_box", "base_head_form": "linear",
                                      "base_head_net": "separate", "base_param_inputs": True,
                                      "init_model_from": path}))
    c, x = codes(4), torch.rand(4, 200, 3, generator=gen()) * 2 - 1
    with torch.no_grad():
        assert torch.allclose(box.model(c, x), sphere.model(c, x), atol=1e-6)
    assert math.isfinite(box.step()["loss"])


def test_config_checks_the_base_network():
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="rounded_box", base_head_net="elsewhere")
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="rounded_box", base_features=False, base_param_inputs=True)


def test_shared_mlp_head_reads_the_side_network_through_a_hidden_layer():
    dec = ModulatedSphereDecoder(linear_cfg(base_head_net="shared_mlp"))
    assert isinstance(dec.base_head_mlp[1], torch.nn.Tanh) and dec.base_head_mlp[0].in_features == 32
    z = codes(6)
    mod = dec.prepare(z)
    assert torch.allclose(mod.base[:, :3], torch.zeros(6, 3), atol=1e-6)
    assert torch.allclose(mod.base[:, 3], torch.full((6,), 0.5), atol=1e-6)
    with torch.no_grad():
        dec.base_head_mlp[-1].weight.normal_(0, 0.5)
    dec.prepare(z).base.sum().backward()
    assert dec.base_head_mlp[0].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.mapping.parameters())   # shared: trains it


# ------------------------------------------------------------------ base variation regularisers
REG = {**TRAIN, "base_shape": "rounded_box", "base_head_form": "linear", "base_head_net": "shared_mlp",
       "grad_clip": 0.0}


def _grads_after_one_step(**changes):
    torch.manual_seed(0)
    trainer = T.Trainer(DataFreeConfig(**{**REG, **changes}))
    with torch.no_grad():                                    # move the base off the sphere so the field pulls on it
        trainer.model.base_head_mlp[-1].weight.normal_(0, 0.02, generator=gen(3))
    trainer.optimizer.step = lambda *a, **k: None            # keep the gradients, change nothing
    record = trainer.step()
    head = torch.cat([p.grad.reshape(-1) for p in trainer.model.base_head_mlp.parameters()])
    main = torch.cat([p.grad.reshape(-1) for p in trainer.model.hidden.parameters()])
    return head, main, record


def test_field_gradient_into_the_base_is_scaled_exactly():
    full_head, full_main, _ = _grads_after_one_step()
    none_head, none_main, record = _grads_after_one_step(base_eikonal_grad_scale=0.0, base_consistency_grad_scale=0.0)
    half_head, half_main, _ = _grads_after_one_step(base_eikonal_grad_scale=0.5, base_consistency_grad_scale=0.5)
    assert "base/field_grad_change" in record and record["base/field_grad_change"] > 0
    size = float(full_head.norm())
    assert size > 1e-6 and float((full_head - none_head).norm()) > 1e-3 * size   # the field terms do pull on the base
    assert float((half_head - 0.5 * (full_head + none_head)).norm()) < 1e-4 * size   # linear in the factor
    assert torch.allclose(full_main, none_main, atol=1e-6) and torch.allclose(full_main, half_main, atol=1e-6)


def test_field_gradient_scaling_with_accumulation():
    wide = dict(global_codes=True, diversity_mode="nearest", accumulate=2, batch_neighbours=True,
                pool_points=40, nearest_k=3)
    trainer = T.Trainer(DataFreeConfig(**{**REG, **wide, "base_consistency_grad_scale": 0.1}))
    record = trainer.step()
    assert math.isfinite(record["loss"]) and record["base/field_grad_change"] > 0


def test_base_spread_constraint_measures_and_pushes_the_spread():
    trainer = T.Trainer(DataFreeConfig(**{**REG, "base_spread_target": 0.1, "base_spread_target_r": 0.05,
                                          "base_spread_mult_init": 100.0}))
    first = trainer.step()
    assert first["base/spread_shortfall"] == pytest.approx(3 * 0.1 ** 2 + 0.05 ** 2, rel=1e-3)   # all bases equal
    assert first["mult/base_spread"] > 100.0                                                        # short: weight grows
    head_grad = [p.grad for p in trainer.model.base_head_mlp.parameters()]
    for _ in range(5):
        record = trainer.step()
    assert record["base/std_b"] > 0 and record["base/spread_shortfall"] < first["base/spread_shortfall"]


def test_base_lr_scale_gives_the_base_head_its_own_rate_on_a_resume(tmp_path):
    cfg = DataFreeConfig(**{**REG, "grad_clip": 1.0})
    trainer = T.Trainer(cfg)
    for _ in range(2):
        trainer.step()
    path = str(tmp_path / "one.pt")
    trainer.save_checkpoint(path)
    saved = {id(p): trainer.optimizer.state[p]["exp_avg"].clone() for p in trainer.model.parameters()}
    names = [n for n, _ in trainer.model.named_parameters()]
    before = dict(zip(names, saved.values()))
    moved = T.Trainer.from_checkpoint(path, overrides=["base_lr_scale=0.1", "out_lr_scale=0.5"], tensorboard=False)
    groups = {g["name"]: g for g in moved.optimizer.param_groups}
    assert set(groups) == {"main", "base", "out"}
    base_ids = {id(p) for p in moved.model.base_head_mlp.parameters()}
    assert {id(p) for p in groups["base"]["params"]} == base_ids
    assert groups["base"]["lr"] == pytest.approx(groups["main"]["lr"] * 0.1)
    assert groups["out"]["lr"] == pytest.approx(groups["main"]["lr"] * 0.5)
    for name, p in moved.model.named_parameters():                 # Adam state carried per parameter
        assert torch.equal(moved.optimizer.state[p]["exp_avg"], before[name])
    moved.step()
    again_path = str(tmp_path / "three.pt")
    moved.save_checkpoint(again_path)
    back = T.Trainer.from_checkpoint(again_path, tensorboard=False)  # same groups by name: direct load
    assert [g["name"] for g in back.optimizer.param_groups] == ["main", "base", "out"]
    plain = T.Trainer.from_checkpoint(again_path, overrides=["base_lr_scale=1.0", "out_lr_scale=1.0"], tensorboard=False)
    assert [g["name"] for g in plain.optimizer.param_groups] == ["main"]
    plain.step()


def test_correction_cost_measures_the_correction_near_the_surface_and_pulls_both_ways():
    cfg = DataFreeConfig(**{**REG, "correction_cost_weight": 25.0, "correction_cost_band": 0.1})
    trainer = T.Trainer(cfg)
    with torch.no_grad():
        trainer.model.base_head_mlp[-1].weight.normal_(0, 0.02, generator=gen(3))
        trainer.model.out.weight.normal_(0, 0.05, generator=gen(4))      # a correction worth measuring
    batch = trainer.sample_batch()
    from arm_planner.data_free.field import evaluate_field
    fb = evaluate_field(trainer.model, batch.codes, batch.cluster, batch.x, cfg)
    value, info = trainer._correction_cost(fb)
    corr = fb.f - rounded_box_field(fb.x, fb.mod.base)
    near = fb.f.detach().abs() < 0.1
    assert float(value) == pytest.approx(float(corr[near].square().mean()), rel=1e-5)
    value.backward()
    assert trainer.model.out.weight.grad.abs().sum() > 0                    # the correction is pulled in
    # f - base is the correction itself: the base is reached only through the base features (zero at start)
    assert trainer.model.base_head_mlp[-1].weight.grad.abs().sum() == 0
    record = trainer.step()
    assert math.isfinite(record["loss"]) and "corr/near_sq" in record


# ------------------------------------------------------------------ bends and tapers
from arm_planner.data_free.decoder import deform_points

DEF = dict(base_head_form="linear", base_head_net="shared_mlp", base_deform=True, base_center_points=4096)


def deform_dec(**kw):
    return ModulatedSphereDecoder(box_cfg(**{**DEF, **kw}))


def set_head(dec, values):
    """Set the head's output for every code: last-layer bias = values, weights zero."""
    with torch.no_grad():
        last = dec.base_head_mlp[-1]
        last.weight.zero_()
        last.bias.copy_(torch.tensor(values, dtype=last.bias.dtype))


def soft_centroid(dec, mod, n=56):
    axis = torch.linspace(-0.9, 0.9, n)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3).expand(mod.base.shape[0], -1, -1)
    w = torch.sigmoid(dec.base_field(mod, grid) / dec.center_tau)
    return (w[..., None] * grid).sum(1) / w.sum(1)[:, None]


def test_zero_deformation_is_the_plain_box():
    dec = deform_dec()
    plain = ModulatedSphereDecoder(box_cfg(base_head_form="linear", base_head_net="shared_mlp"))
    vals = [0.2, 0.05, -0.02, -0.25] + [0.0] * 9
    set_head(dec, vals)
    set_head(plain, vals[:4])
    z, x = codes(3), torch.rand(3, 500, 3, generator=gen()) * 2 - 1
    with torch.no_grad():
        mod = dec.prepare(z)
        assert torch.allclose(mod.deform[:, 9:], torch.zeros(3, 3), atol=1e-6)     # antithetic points: no shift
        assert torch.allclose(dec.base_field(mod, x), plain.base_field(plain.prepare(z), x), atol=1e-5)


def test_bends_and_tapers_are_recentred():
    dec = deform_dec()
    for vals in ([0.3, 0.05, 0.02, -0.38, 1.5] + [0.0] * 8,                  # rod along x bent toward y
                 [0.25, 0.2, 0.0, -0.4, 0, 0, 0, 0, -1.2, 0.0, 0.0, 0.8, 0.0],  # plate, bent and tapered
                 [0.3, 0.0, 0.0, -0.35, 0, 0, 0, 0, 0, 0, 0.9, 0.0, 0.0]):     # rod tapered along x
        set_head(dec, vals)
        with torch.no_grad():
            mod = dec.prepare(codes(1))
            raw = mod._replace(deform=torch.cat([mod.deform[:, :9], torch.zeros(1, 3)], -1))
            before = soft_centroid(dec, raw)
            after = soft_centroid(dec, mod)
        assert float(before.norm()) > 0.01                       # the deformation moved the centre of mass
        assert float(after.norm()) < 0.15 * float(before.norm()) + 2e-3   # recentring puts it back


def test_deformed_base_stays_inside_the_reach_ball_and_the_outer_bound_holds():
    cfg = box_cfg(**DEF, output_init_std=1.0)
    dec = ModulatedSphereDecoder(cfg)
    set_head(dec, [0.5, 0.3, 0.2, -0.4] + [5.0, -5.0, 5.0, 5.0, -5.0, 5.0, 3.0, -3.0, 3.0])   # extreme
    x = torch.rand(4, 6000, 3, generator=gen(5)) * 2 - 1
    with torch.no_grad():
        mod = dec.prepare(codes(4))
        assert float(mod.deform[:, :6].abs().max()) <= cfg.base_bend_max + 1e-6
        base = dec.base_field(mod, x)
        f = dec.query(mod, x)
    far = x.norm(dim=-1) > cfg.max_reach()
    assert (base[far] <= 0).all()
    assert (f[x.norm(dim=-1) > cfg.bound_radius] < 0).all()


def test_deformation_changes_smoothly_also_where_extents_cross():
    dec = deform_dec()
    x = torch.rand(1, 3000, 3, generator=gen(6)) * 1.2 - 0.6
    fields = []
    for t in torch.linspace(-0.02, 0.02, 5):                     # extents of x and y cross at t = 0
        set_head(dec, [0.2 + float(t), 0.2 - float(t), 0.0, -0.3, 1.0, 0.5, -0.5, 0.8, 0.0, 0.0, 0.3, -0.3, 0.0])
        with torch.no_grad():
            fields.append(dec.base_field(dec.prepare(codes(1)), x))
    steps = [float((b - a).abs().max()) for a, b in zip(fields, fields[1:])]
    assert max(steps) < 0.03 and max(steps) < 2.0 * min(steps) + 1e-3    # even steps: no jump at the crossing


def test_gradient_reaches_the_deformation_outputs_through_the_centring():
    dec = deform_dec()
    set_head(dec, [0.25, 0.05, 0.0, -0.35, 0.5, 0, 0, 0, 0, 0, 0.3, 0, 0])
    mod = dec.prepare(codes(2))
    x = torch.rand(2, 400, 3, generator=gen(7)) * 1.4 - 0.7
    dec.base_field(mod, x).square().mean().backward()
    g = dec.base_head_mlp[-1].bias.grad
    assert (g[4:13].abs() > 0).sum() >= 6                       # bends and tapers all get a gradient
    mod2 = dec.prepare(codes(2))
    (gc,) = torch.autograd.grad(mod2.deform[:, 9:].sum(), dec.base_head_mlp[-1].bias)
    assert gc[4].abs() > 0                                       # the centring shift depends on the bend


def test_deform_trainer_step_with_spread_push_on_the_new_parameters():
    cfg = DataFreeConfig(**{**TRAIN, "base_shape": "rounded_box", **DEF, "base_spread_target": 0.1,
                            "base_spread_target_r": 0.05, "base_spread_target_bend": 1.0,
                            "base_spread_target_taper": 0.3, "correction_cost_weight": 50.0})
    trainer = T.Trainer(cfg)
    record = trainer.step()
    assert math.isfinite(record["loss"])
    assert "base/std_bend" in record and "base/std_taper" in record
    assert record["base/spread_shortfall"] == pytest.approx(3 * 0.01 + 0.0025 + 6 * 1.0 + 3 * 0.09, rel=1e-3)
    out = trainer.run_diagnostics()
    assert "base/bend_abs_mean" in out and "base/centroid_residual_mean" in out


def test_config_checks_the_deformations():
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="rounded_box", base_deform=True)                       # needs the linear head
    with pytest.raises(ValueError):
        DataFreeConfig(base_shape="rounded_box", base_head_form="linear", base_deform=True, base_surface_points=8)


def test_resume_into_bends_and_tapers_keeps_the_field_and_bounds_the_spread_push(tmp_path):
    base_cfg = {**TRAIN, "base_shape": "rounded_box", "base_head_form": "linear", "base_head_net": "shared_mlp",
                "base_spread_target": 0.1, "base_spread_target_r": 0.05, "base_lr_scale": 0.1,
                "correction_cost_weight": 50.0}
    old = T.Trainer(DataFreeConfig(**base_cfg), output_dir=str(tmp_path), tensorboard=False)
    with torch.no_grad():
        old.model.base_head_mlp[-1].weight.normal_(0, 0.05, generator=gen(8))
    for _ in range(3):
        old.step()
    old.save_checkpoint()
    path = str(tmp_path / T.CHECKPOINT_NAME)
    new = T.Trainer.from_checkpoint(path, tensorboard=False, overrides=[
        "base_deform=True", "base_spread_target_bend=1.0", "base_spread_target_taper=0.3",
        "base_spread_mult_max=100", "base_spread_tolerance=1e-3", "base_spread_ramp_from=3",
        "base_spread_ramp_steps=10", "base_center_points=4096"])
    last_old, last_new = old.model.base_head_mlp[-1], new.model.base_head_mlp[-1]
    assert last_new.weight.shape[0] == 13
    assert torch.equal(last_new.weight[:4], last_old.weight) and (last_new.weight[4:] == 0).all()
    c, x = codes(4), torch.rand(4, 400, 3, generator=gen()) * 2 - 1
    with torch.no_grad():
        assert torch.allclose(new.model(c, x), old.model(c, x), atol=1e-5)     # same field at the start
    assert last_new.weight not in new.optimizer.state                        # grown: fresh Adam state
    first = new.model.hidden[0].weight
    assert torch.equal(new.optimizer.state[first]["exp_avg"], old.optimizer.state[old.model.hidden[0].weight]["exp_avg"])
    assert new.multipliers.range("base_spread")[1] == 100.0
    assert new.multipliers.tolerance("base_spread") == pytest.approx(1e-3)
    record = new.step()                                                       # at step 3: ramp 0, targets 0
    assert math.isfinite(record["loss"])
    for _ in range(5):
        record = new.step()                                                   # ramp half way at step 8
    assert new.multipliers.multiplier("base_spread") <= 100.0
    assert record["base/spread_shortfall"] <= 6 * 0.5 ** 2 + 3 * 0.15 ** 2 + 0.03       # targets still ramping


# ------------------------------------------------------------------ base repulsion (main diversity mechanism)
from arm_planner.data_free import losses as L


def test_base_repel_matches_the_nearest_neighbour_formula():
    cfg = DataFreeConfig(nearest_k=2)
    v = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]], requires_grad=True)
    loss, info = L.base_repel(v, v.new_zeros((0, 2)), cfg)
    d = torch.cdist(v.detach(), v.detach(), p=1) / 2
    d.fill_diagonal_(float("inf"))
    near = d.topk(2, dim=1, largest=False).values
    soft = (near + 1e-6).pow(-4).sum(1).pow(-0.25)
    assert float(loss) == pytest.approx(float((-torch.log(soft + 1e-3)).mean()), rel=1e-5)
    loss.backward()
    assert v.grad.abs().sum() > 0


def test_base_repel_scores_two_groups_worse_than_an_even_spread():
    cfg = DataFreeConfig(nearest_k=4)
    g = gen(9)
    groups = torch.cat([torch.full((16, 1), -1.0), torch.full((16, 1), 1.0)]) + 0.01 * torch.randn(32, 1, generator=g)
    even = torch.linspace(-1, 1, 32)[:, None]
    even = even * (groups.std() / even.std())                    # same standard deviation
    lg, _ = L.base_repel(groups, groups.new_zeros((0, 1)), cfg)
    le, _ = L.base_repel(even, even.new_zeros((0, 1)), cfg)
    assert float(lg) > float(le) + 1.0                           # the std push could not tell these apart


def test_base_repel_with_the_whole_update_and_a_symmetry_breaking_start(tmp_path):
    wide = dict(global_codes=True, diversity_mode="nearest", accumulate=2, batch_neighbours=True,
                pool_points=40, nearest_k=3)
    base_cfg = {**TRAIN, **wide, "base_shape": "rounded_box", "base_head_form": "linear", "base_head_net": "shared_mlp"}
    old = T.Trainer(DataFreeConfig(**base_cfg), output_dir=str(tmp_path), tensorboard=False)
    old.step()
    old.save_checkpoint()
    new = T.Trainer.from_checkpoint(str(tmp_path / T.CHECKPOINT_NAME), tensorboard=False, overrides=[
        "base_deform=True", "base_center_points=2048", "base_repel_weight=0.5", "base_deform_init_std=1e-3"])
    rows = new.model.base_head_mlp[-1].weight[4:]
    assert 0 < float(rows.abs().max()) < 0.01                    # tiny random start of the new rows
    c, x = codes(4), torch.rand(4, 300, 3, generator=gen()) * 2 - 1
    with torch.no_grad():
        assert float((new.model(c, x) - old.model(c, x)).abs().max()) < 0.01
    record = new.step()
    assert math.isfinite(record["loss"]) and "base_repel/term" in record
    assert new._shared_base is None                              # cleared after the update
    with torch.no_grad():
        bend = new.model.prepare(codes(64)).deform[:, :6]
    assert float(bend.std(dim=0).mean()) > 0                     # bends differ between codes
