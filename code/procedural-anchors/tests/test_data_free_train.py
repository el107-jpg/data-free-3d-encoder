"""Tests for arm_planner.data_free.train (CPU, tiny config)."""

import dataclasses
import json
import math
import os

import pytest
import torch

from arm_planner.data_free import train as T
from arm_planner.data_free.config import DataFreeConfig

SMALL = dict(
    width=32, layers=3, mapping_width=32, clusters=2, codes_per_cluster=3, points=256,
    bank_size=16, hard_negatives=8, grad_change_points=32, steps=20, lr_warmup_steps=5,
    diag_every=0, ckpt_every=0, probe_codes=4, diag_resolution=16,
    dead_direction_probes=1, dead_direction_points=512,
)
SMALL_SET = [f"{k}={v}" for k, v in SMALL.items()]


def small_cfg(**changes):
    return DataFreeConfig(**{**SMALL, **changes})


def assert_finite(record):
    for key, value in record.items():
        if isinstance(value, str):
            continue
        assert math.isfinite(value), f"{key} = {value}"


# ------------------------------------------------------------------- the step
def test_steps_are_finite_and_parameters_change():
    trainer = T.Trainer(small_cfg())
    before = [p.detach().clone() for p in trainer.model.parameters()]
    records = [T.train_step(trainer) for _ in range(4)]
    for record in records:
        assert_finite(record)
        assert record["skipped"] == 0
        assert record["graph_mode"] in T.GRAPH_MODES
        for key in T.CONSTRAINT_KEYS:
            assert f"term/{key}" in record and f"mult/{key}" in record
    assert [r["step"] for r in records] == [0, 1, 2, 3]
    assert trainer.step_count == 4
    changed = [not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters())]
    assert all(changed)


def test_same_seed_same_run_and_other_seed_differs():
    a = [T.Trainer(small_cfg()).step()["loss"] for _ in range(1)]
    b = [T.Trainer(small_cfg()).step()["loss"] for _ in range(1)]
    c = [T.Trainer(small_cfg(seed=1)).step()["loss"] for _ in range(1)]
    assert a == b
    assert a != c


def test_multiplier_of_an_impossible_tolerance_rises():
    cfg = small_cfg(mult_min=1e-2, mult_max=1e4)
    cfg.tolerances["eikonal"] = -1.0       # never met
    cfg.tolerances["consistency"] = 1e9    # always met
    trainer = T.Trainer(cfg)
    for _ in range(5):
        record = trainer.step()
    assert record["mult/eikonal"] == pytest.approx(cfg.mult_init * (1 + cfg.mult_rate) ** 5)
    assert record["mult/consistency"] == pytest.approx(cfg.mult_init * (1 - cfg.mult_rate / 2) ** 5)


def test_multipliers_stay_inside_the_default_range():
    cfg = small_cfg()
    assert (cfg.mult_min, cfg.mult_max) == (1.0, 200.0)
    cfg.tolerances["eikonal"] = -1.0       # never met: runs to the cap
    cfg.mult_rate = 1.0                    # x2 per violated step
    trainer = T.Trainer(cfg)
    for _ in range(8):
        record = trainer.step()
    assert record["mult/eikonal"] == 200.0
    assert all(record[f"mult/{k}"] >= 1.0 for k in T.CONSTRAINT_KEYS)


def test_trainer_uses_per_constraint_ranges():
    cfg = small_cfg(mult_rate=1.0)
    cfg.mult_rates = {}
    for key in ("pose_centroid", "barrier", "eikonal"):
        cfg.tolerances[key] = -1.0         # never met
    trainer = T.Trainer(cfg)
    for _ in range(15):
        record = trainer.step()
    assert record["mult/pose_centroid"] == 50.0
    assert record["mult/eikonal"] == 200.0
    assert record["mult/barrier"] == 1e4


def test_pose_enters_as_three_separate_constraints():
    assert T.FIELD_KEYS[-3:] == ("pose_centroid", "pose_scale", "pose_axes")
    assert "pose" not in T.CONSTRAINT_KEYS
    cfg = small_cfg()
    for key in T.FIELD_KEYS[-3:]:
        assert key in cfg.tolerances
    record = T.Trainer(cfg).step()
    assert "term/pose" not in record and "mult/pose" not in record
    for key in T.FIELD_KEYS[-3:]:
        assert math.isfinite(record[f"term/{key}"]) and record[f"mult/{key}"] > 0


def test_loss_is_objective_plus_weighted_constraints():
    cfg = small_cfg()
    trainer = T.Trainer(cfg)
    record = trainer.step()
    total = record["diversity"] + cfg.grad_change_weight * record["grad_change"]
    total += sum(record[f"mult/{k}"] * record[f"term/{k}"] for k in T.CONSTRAINT_KEYS)
    assert record["loss"] == pytest.approx(total, rel=1e-4, abs=1e-7)


def test_graph_is_built_once_per_batch(monkeypatch):
    calls = []
    real = T.build_graph

    def counting(*args, **kwargs):
        calls.append(kwargs.get("mode"))
        return real(*args, **kwargs)

    monkeypatch.setattr(T, "build_graph", counting)
    trainer = T.Trainer(small_cfg(graph_reuse_steps=3))
    records = [trainer.step() for _ in range(6)]
    assert all(r["graph_mode"] != "off" for r in records)
    assert len(calls) == 2
    assert [r["graph_built"] for r in records] == [1, 0, 0, 1, 0, 0]
    # same codes and points within a batch, new ones for the next batch
    trainer = T.Trainer(small_cfg(graph_reuse_steps=2))
    trainer.step()
    first = trainer.batch
    trainer.step()
    assert trainer.batch is first and first.inner == 2
    trainer.step()
    assert trainer.batch is not first
    assert not torch.equal(trainer.batch.x, first.x)


def test_graph_mode_thresholds():
    cfg = small_cfg(graph_sign_error=0.05, graph_full_error=0.02)
    assert T.graph_mode_for(0.06, cfg) == "off"
    assert T.graph_mode_for(0.03, cfg) == "sign"
    assert T.graph_mode_for(0.01, cfg) == "full"
    assert T.graph_mode_for(float("nan"), cfg) == "off"


def test_graph_off_leaves_action_multipliers_alone(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("build_graph called while the graph is off")

    monkeypatch.setattr(T, "build_graph", forbidden)
    trainer = T.Trainer(small_cfg(graph_sign_error=-1.0, mult_min=1e-2))
    record = trainer.step()
    assert record["graph_mode"] == "off"
    for key in T.ACTION_KEYS:
        assert record[f"mult/{key}"] == trainer.cfg.mult_init
        assert record[f"term/{key}"] == 0.0
    assert record["mult/eikonal"] != trainer.cfg.mult_init


# ---------------------------------------------------------------- anchor bank
def test_batch_layout_and_bank_fraction():
    cfg = small_cfg(clusters=4, bank_fraction=0.5)
    trainer = T.Trainer(cfg)
    batch = trainer.sample_batch()
    assert batch.codes.shape == (12, cfg.latent_dim)
    assert batch.x.shape == (4, 256, 3)
    assert torch.allclose(batch.codes.norm(dim=-1), torch.ones(12), atol=1e-5)
    assert (batch.bank_index[:2] >= 0).all() and (batch.bank_index[2:] == -1).all()
    centres = batch.codes[::3]
    assert torch.allclose(centres[:2], trainer.bank.codes[batch.bank_index[:2]], atol=1e-6)
    assert (batch.eps >= cfg.eps_min - 1e-6).all() and (batch.eps <= cfg.eps_max + 1e-6).all()
    from arm_planner.data_free.latent import chord
    spread = chord(batch.codes.view(4, 3, -1), centres[:, None, :])
    assert (spread <= batch.eps[:, None] + 1e-4).all()
    assert (batch.x[:, 0] == 0).all()


def test_hard_negatives_are_read_back_for_the_same_anchor():
    cfg = small_cfg(bank_size=2, bank_fraction=1.0)
    trainer = T.Trainer(cfg)
    gen = torch.Generator().manual_seed(3)
    points = torch.full((1, cfg.hard_negatives, 3), float("nan"))
    points[0, :5] = torch.rand(5, 3, generator=gen) - 0.5
    trainer.bank.write(torch.tensor([1]), points)
    batch = trainer.sample_batch()
    assert sorted(batch.bank_index.tolist()) == [0, 1]
    row = int((batch.bank_index == 1).nonzero())
    other = 1 - row

    def contains(x, p):
        return bool(((x - p).abs().amax(dim=-1) < 1e-6).any())

    for p in points[0, :5]:
        assert contains(batch.x[row], p)
        assert not contains(batch.x[other], p)
    # they sit in the tail rows, in cache order, and the origin row is intact
    tail = batch.x[row, cfg.points - cfg.hard_negatives:]
    assert torch.allclose(tail[:5], points[0, :5])
    assert (batch.x[:, 0] == 0).all()


def test_bank_write_merges_new_first_and_drops_repeats():
    cfg = small_cfg(hard_negatives=4)
    trainer = T.Trainer(cfg)
    bank = trainer.bank
    nan = float("nan")
    a, b, c, d, e = (torch.full((3,), v) for v in (0.1, 0.2, 0.3, 0.4, 0.5))
    pad = torch.full((3,), nan)
    bank.write(torch.tensor([5]), torch.stack([a, b, pad, pad])[None])
    assert torch.allclose(bank.read(torch.tensor([5]))[0, :2], torch.stack([a, b]))
    assert torch.isnan(bank.read(torch.tensor([5]))[0, 2:]).all()
    # b comes back as new: kept once; c is new; a survives behind them
    bank.write(torch.tensor([5]), torch.stack([c, b, pad, pad])[None])
    assert torch.allclose(bank.read(torch.tensor([5]))[0, :3], torch.stack([c, b, a]))
    assert torch.isnan(bank.read(torch.tensor([5]))[0, 3]).all()
    # overflow: newest win
    bank.write(torch.tensor([5]), torch.stack([d, e, pad, pad])[None])
    assert torch.allclose(bank.read(torch.tensor([5]))[0], torch.stack([d, e, c, b]))
    assert torch.isnan(bank.read(torch.tensor([4]))).all()


def test_step_writes_graph_hard_negatives_to_the_bank(monkeypatch):
    cfg = small_cfg(bank_size=2, bank_fraction=1.0)
    trainer = T.Trainer(cfg)
    marker = torch.tensor([0.25, -0.125, 0.0625])

    def fake(graph, fb, cfg_, per_shape, generator=None):
        out = torch.full((fb.x.shape[0], per_shape, 3), float("nan"))
        out[:, 0] = marker                                  # every code reports one point
        out[::cfg.codes_per_cluster, 1] = -marker           # only centres report a second
        return out

    monkeypatch.setattr(T, "hard_negatives", fake)
    trainer.step()
    cache = trainer.bank.negatives
    assert torch.allclose(cache[:, 0], marker.expand(2, 3))
    assert torch.allclose(cache[:, 1], -marker.expand(2, 3))
    assert torch.isnan(cache[:, 2:]).all()
    trainer.step()  # reuse step: no second write
    trainer.step()  # next batch reads them back
    tail = trainer.batch.x[:, cfg.points - cfg.hard_negatives:]
    assert torch.allclose(tail[:, 0], marker.expand(2, 3))
    assert torch.allclose(tail[:, 1], -marker.expand(2, 3))


# ---------------------------------------------------------------- checkpoints
@pytest.mark.parametrize("steps_before", [2, 3])  # at a batch boundary and inside a batch
def test_checkpoint_resume_reproduces_the_next_steps(tmp_path, steps_before):
    trainer = T.Trainer(small_cfg())
    for _ in range(steps_before):
        trainer.step()
    path = trainer.save_checkpoint(str(tmp_path / "ckpt.pt"))
    expected = [trainer.step() for _ in range(3)]

    resumed = T.Trainer.from_checkpoint(path)
    assert resumed.step_count == steps_before
    got = [resumed.step() for _ in range(3)]
    for a, b in zip(expected, got):
        for key, value in a.items():
            if key.startswith("time_ms/") or key in ("gpu_peak_mb", "graph/build_ms"):
                continue
            if isinstance(value, str):
                assert b[key] == value
            else:
                assert b[key] == pytest.approx(value, rel=1e-5, abs=1e-8), key
    assert torch.equal(resumed.bank.codes, trainer.bank.codes)
    assert torch.equal(torch.isnan(resumed.bank.negatives), torch.isnan(trainer.bank.negatives))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_checkpoint_resume_on_gpu(tmp_path):
    trainer = T.Trainer(small_cfg(), device="cuda")
    for _ in range(3):
        trainer.step()
    path = trainer.save_checkpoint(str(tmp_path / "ckpt.pt"))
    expected = trainer.step()
    resumed = T.Trainer.from_checkpoint(path, device="cuda")
    got = resumed.step()
    assert got["graph_mode"] == expected["graph_mode"]
    assert got["loss"] == pytest.approx(expected["loss"], rel=1e-3)
    assert got["div/mean_ratio"] == pytest.approx(expected["div/mean_ratio"], rel=1e-3)


def test_checkpoint_holds_the_full_state(tmp_path):
    trainer = T.Trainer(small_cfg())
    trainer.step()
    state = torch.load(trainer.save_checkpoint(str(tmp_path / "c.pt")), weights_only=False)
    for key in ("config", "step", "model", "optimizer", "scheduler", "multipliers", "bank",
                "generator", "torch_rng", "batch"):
        assert key in state
    assert state["step"] == 1
    assert state["bank"]["negatives"].shape == (16, 8, 3)


# ------------------------------------------------------------------- schedule
def test_lr_schedule():
    cfg = small_cfg(steps=100, lr_warmup_steps=10, lr_floor_fraction=0.1)
    assert T.lr_factor(0, cfg) == pytest.approx(0.1)
    assert T.lr_factor(9, cfg) == pytest.approx(1.0)
    assert T.lr_factor(10, cfg) == pytest.approx(1.0)
    assert T.lr_factor(55, cfg) == pytest.approx(0.55)
    assert T.lr_factor(100, cfg) == pytest.approx(0.1)
    assert T.lr_factor(500, cfg) == pytest.approx(0.1)
    # a decay horizon shorter than the run holds the rate at the floor for the rest of it
    held = small_cfg(steps=400, lr_warmup_steps=10, lr_floor_fraction=0.5, lr_decay_steps=100)
    assert T.lr_factor(55, held) == pytest.approx(0.75)
    assert T.lr_factor(100, held) == T.lr_factor(399, held) == pytest.approx(0.5)
    trainer = T.Trainer(small_cfg(lr=1e-3, lr_warmup_steps=4))
    lrs = [trainer.step()["lr"] for _ in range(5)]
    assert lrs[:4] == pytest.approx([0.25e-3, 0.5e-3, 0.75e-3, 1e-3])
    assert lrs[4] == pytest.approx(1e-3, rel=1e-2)


# ------------------------------------------------------------------ overrides
def test_set_type_conversion():
    assert T.parse_override("points=512") == ("points", 512)
    assert isinstance(T.parse_override("points=512")[1], int)
    assert T.parse_override("lr=1e-4") == ("lr", 1e-4)
    assert T.parse_override("steps=2e3") == ("steps", 2000)
    assert T.parse_override("use_window=true") == ("use_window", True)
    assert T.parse_override("graph_debug=0") == ("graph_debug", False)
    assert T.parse_override("surface_edge_radius=none") == ("surface_edge_radius", None)
    assert T.parse_override("surface_edge_radius=0.2") == ("surface_edge_radius", 0.2)
    assert T.parse_override("tolerances.eikonal=0.5") == ("tolerances.eikonal", 0.5)
    assert T.parse_override("mult_rates.delete=0.1") == ("mult_rates.delete", 0.1)
    assert T.parse_override("mult_ranges.delete=1,5e3") == ("mult_ranges.delete", (1.0, 5000.0))
    changed = T.apply_overrides(DataFreeConfig(), ["mult_ranges.pose_scale=2,30", "mult_rates.fill=0.2"])
    assert tuple(changed.mult_ranges["pose_scale"]) == (2.0, 30.0)
    assert tuple(changed.mult_ranges["delete"]) == (1.0, 1e4) and changed.mult_rates["fill"] == 0.2
    with pytest.raises(KeyError):
        T.parse_override("nonsense.delete=1")
    assert T.parse_override('tolerances={"pose": 1}') == ("tolerances", {"pose": 1})
    with pytest.raises(KeyError):
        T.parse_override("no_such_field=1")
    with pytest.raises(ValueError):
        T.parse_override("points=1.5")
    with pytest.raises(ValueError):
        T.parse_override("use_window=maybe")

    cfg = T.apply_overrides(DataFreeConfig(), ["width=64", "tolerances.pose=0.5", "min_feature=0.1"])
    assert cfg.width == 64 and cfg.tolerances["pose"] == 0.5
    assert cfg.tolerances["eikonal"] == DataFreeConfig().tolerances["eikonal"]
    assert cfg.min_feature == 0.1
    with pytest.raises(ValueError):
        T.apply_overrides(DataFreeConfig(), ["correction_bound=0.6"])  # config validation runs


def test_config_has_the_graph_knobs_with_graph_defaults():
    cfg = DataFreeConfig()
    assert cfg.graph_one_sided_cos == 0.3 and cfg.graph_one_sided_cut is True
    assert cfg.graph_hidden_gap_drop is True and cfg.graph_kind1_core is None
    assert cfg.graph_rescue_degree == 4 and cfg.graph_rescue_cos == 0.05
    assert cfg.graph_gap_resolution is None and cfg.graph_debug is False
    assert cfg.descriptor_weight == 0.02


# ---------------------------------------------------------------- diagnostics
def test_diagnostics_are_finite_and_leave_training_untouched():
    trainer = T.Trainer(small_cfg())
    trainer.step()
    reference = T.Trainer(small_cfg())
    reference.step()
    record = trainer.run_diagnostics()
    for key in ("grid/clean_fraction", "grid/one_piece_fraction", "grid/handles_mean",
                "graph_vs_grid/both_agreement", "dead/live_mean", "dead/wobble_share_mean",
                "descriptor_spread", "probe_field_distance", "e",
                "graph_vs_grid/confirm_dropped_kind0", "graph_vs_grid/confirm_dropped_cavities",
                "graph_vs_grid/raw_cavities", "dead/subspace_stability_mean",
                "dead/subspace_stability_count", "pose/centroid_offset_max",
                "pose/trace_rel_error_max", "pose/offdiag_abs_max"):
        assert key in record
    assert_finite(record)
    # a fresh decoder is a sphere for every code
    assert record["grid/clean_fraction"] == 1.0
    # ... centred and of the target size, as measured with many points
    assert record["pose/centroid_offset_max"] < 0.01
    assert record["pose/trace_rel_error_max"] < 0.03
    assert trainer.model.training
    assert trainer.step()["loss"] == reference.step()["loss"]


def test_diagnostics_measure_a_planted_pose_error():
    trainer = T.Trainer(small_cfg(diag_pose_points=200_000))
    gen = torch.Generator().manual_seed(0)
    codes = trainer.probe_codes()

    class Shifted(torch.nn.Module):
        def prepare(self, c):
            return c

        def query(self, mod, x):
            return 0.6 - (x - torch.tensor([0.05, 0.0, 0.0])).norm(dim=-1)

    trainer.model = Shifted()
    pose = trainer._probe_pose(codes, gen)
    assert pose["pose/centroid_offset_mean"] == pytest.approx(0.05, abs=0.004)
    from arm_planner.data_free.losses import soft_ball_trace
    expected = soft_ball_trace(0.6, trainer.cfg.soft_tau) / trainer.cfg.cov_trace_target - 1
    assert pose["pose/trace_rel_error_rms"] == pytest.approx(expected, abs=0.02)
    assert pose["pose/offdiag_abs_max"] < 2e-3


# ------------------------------------------------------------------------ CLI
def test_cli_runs_two_steps_then_resumes(tmp_path):
    out = str(tmp_path / "run")
    args = ["--output-dir", out, "--device", "cpu", "--steps", "2", "--seed", "5", "--quiet"]
    for item in SMALL_SET + ["diag_every=2", "log_every=1"]:
        args += ["--set", item]
    trainer = T.main(args)
    assert trainer.step_count == 2 and trainer.cfg.seed == 5 and trainer.cfg.width == 32
    lines = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert [r["step"] for r in lines] == [0, 1]
    for record in lines:
        assert_finite(record)
    diags = [json.loads(line) for line in open(os.path.join(out, "diagnostics.jsonl"))]
    assert [d["step"] for d in diags] == [0, 2]
    assert os.path.exists(os.path.join(out, T.CHECKPOINT_NAME))
    assert os.path.exists(os.path.join(out, "config.json"))
    assert any(name.startswith("events") for name in os.listdir(os.path.join(out, "tb")))

    resumed = T.main(["--output-dir", out, "--device", "cpu", "--resume", "--steps", "4", "--quiet",
                      "--set", "diag_every=0"])
    assert resumed.step_count == 4 and resumed.cfg.width == 32
    lines = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert [r["step"] for r in lines] == [0, 1, 2, 3]


def test_cli_script_runs_as_a_subprocess(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = str(tmp_path / "run")
    cmd = [sys.executable, str(root / "scripts" / "train_data_free_latent.py"), "--output-dir", out,
           "--device", "cpu", "--steps", "2", "--no-tensorboard"]
    for item in SMALL_SET + ["log_every=1"]:
        cmd += ["--set", item]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root), timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.count("step ") == 2
    assert os.path.exists(os.path.join(out, T.CHECKPOINT_NAME))


def test_global_codes_share_points_and_train():
    """global_codes: independent random codes, one cluster, one shared set of points."""
    trainer = T.Trainer(small_cfg(global_codes=True, diversity_mode="nearest", clusters=2, codes_per_cluster=3))
    batch = trainer.sample_batch()
    assert batch.codes.shape[0] == 6 and int(batch.cluster.max()) == 0 and batch.x.shape[0] == 1
    cos = batch.codes @ batch.codes.T
    assert float((cos - torch.eye(6)).abs().max()) < 0.9          # not a tight cluster
    record = trainer.step()
    assert_finite(record)
    assert record["div/nearest_distance"] >= 0.0


def test_pooled_neighbour_search_trains_and_uses_the_pool():
    cfg = small_cfg(global_codes=True, diversity_mode="nearest", clusters=2, codes_per_cluster=3,
                    pool_size=24, pool_points=64, nearest_k=4, accumulate=2)
    trainer = T.Trainer(cfg)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    record = trainer.step()
    assert_finite(record)
    assert record["div/nearest_distance"] >= 0.0
    assert 0.0 < trainer._pool_share <= 1.0            # some chosen neighbours come from the pool
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))


def test_pooled_mode_skips_the_in_batch_diversity_pass(monkeypatch):
    """With a pool, the loss and the logged values come from nearest_pool alone."""
    calls = {"diversity": 0}
    real = T.L.diversity

    def counted(*args, **kwargs):
        calls["diversity"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(T.L, "diversity", counted)
    pooled = dict(global_codes=True, diversity_mode="nearest", clusters=2, codes_per_cluster=3,
                  pool_size=24, pool_points=64, nearest_k=4)
    for extra in ({}, {"accumulate": 2}):
        calls["diversity"] = 0
        trainer = T.Trainer(small_cfg(**pooled, **extra))
        record = trainer.step()
        assert calls["diversity"] == 0
        assert_finite(record)
        for key in ("mean_ratio", "dead_zone_fraction", "mean_shape_distance", "mean_latent_distance",
                    "nearest_distance", "phase2", "order_violations"):
            assert f"div/{key}" in record
        assert record["div/mean_latent_distance"] > 0.0 and record["div/mean_shape_distance"] >= 0.0
        assert 0.0 <= record["div/dead_zone_fraction"] <= 1.0
        assert 0.0 < trainer._pool_share <= 1.0
        assert isinstance(T.summary_line(record), str)
    # without a pool (and in every other mode) the in-batch pass still runs
    for changes in ({**pooled, "pool_size": 0}, {}):
        calls["diversity"] = 0
        assert_finite(T.Trainer(small_cfg(**changes)).step())
        assert calls["diversity"] >= 1


def test_batch_wide_neighbours_share_points_across_passes(monkeypatch):
    cfg = small_cfg(global_codes=True, diversity_mode="nearest", clusters=2, codes_per_cluster=3,
                    accumulate=3, batch_neighbours=True, pool_points=40, nearest_k=5)
    trainer = T.Trainer(cfg)
    seen = []
    real = T.evaluate_field

    def recording(model, codes, cluster, x, cfg_, *args, **kwargs):
        seen.append(x.detach().clone())
        return real(model, codes, cluster, x, cfg_, *args, **kwargs)

    monkeypatch.setattr(T, "evaluate_field", recording)
    record = trainer.step()
    assert_finite(record)
    assert len(seen) == 3
    for x in seen[1:]:
        assert torch.equal(x[:, 1:41], seen[0][:, 1:41])          # the shared block
        assert not torch.equal(x[:, 41:], seen[0][:, 41:])        # the rest is each pass's own
    assert trainer._pool_share > 0.3                               # most candidates are in other passes
    assert trainer._shared_neighbours is None                      # cleared after the step


def test_accumulated_step_averages_fresh_batches(monkeypatch):
    """accumulate=3: three fresh batches and graph builds, one optimiser step."""
    trainer = T.Trainer(small_cfg(accumulate=3))
    calls = {"graph": 0, "batch": 0}
    real_build, real_sample = T.build_graph, trainer.sample_batch

    def counted_build(*args, **kwargs):
        calls["graph"] += 1
        return real_build(*args, **kwargs)

    def counted_sample():
        calls["batch"] += 1
        return real_sample()

    monkeypatch.setattr(T, "build_graph", counted_build)
    trainer.sample_batch = counted_sample
    before = [p.detach().clone() for p in trainer.model.parameters()]
    record = trainer.step()
    assert calls["batch"] == 3 and calls["graph"] == 3
    assert trainer.step_count == 1 and record["skipped"] == 0
    assert_finite(record)
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))
    trainer.step()
    assert calls["batch"] == 6 and trainer.step_count == 2


# ------------------------------------------------------------ weak-direction pairs
WIDE = dict(global_codes=True, diversity_mode="nearest", clusters=2, codes_per_cluster=4, accumulate=2,
            batch_neighbours=True, pool_points=40, nearest_k=3, weak_probe_codes=6, weak_probe_points=128)


def test_weak_pairs_are_off_by_default_and_need_the_batch_wide_mode():
    trainer = T.Trainer(small_cfg(**WIDE))
    batch = trainer.sample_batch()
    assert batch.pair_first is None and batch.pair_second is None and trainer._usage is None
    with pytest.raises(ValueError):
        T.Trainer(small_cfg(weak_pairs=1))


def test_weak_partners_keep_the_used_part_and_sit_one_chord_away():
    trainer = T.Trainer(small_cfg(**WIDE, weak_pairs=2, weak_chord=0.25, weak_floor=0.04))
    usage = torch.zeros(32, 32, dtype=torch.float64)
    usage[0, 0] = usage[1, 1] = usage[2, 2] = 1.0                   # the decoder "reads" three coordinates
    trainer._set_usage(usage)
    assert trainer._weak_basis.shape == (32, 29)
    assert trainer._usage_info["weak/used_directions"] == 3.0
    batch = trainer.sample_batch()
    assert batch.codes.shape == (8, 32)
    assert batch.pair_first.tolist() == [0, 1] and batch.pair_second.tolist() == [6, 7]
    base, partner = batch.codes[batch.pair_first], batch.codes[batch.pair_second]
    assert torch.allclose(partner[:, :3], base[:, :3], atol=1e-6)    # used part untouched
    assert torch.allclose(partner.norm(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose((base - partner).norm(dim=-1), torch.full((2,), 0.25), atol=1e-5)
    assert torch.allclose(batch.pair_chord, torch.full((2,), 0.25), atol=1e-5)
    # with fewer than two unused directions there is nothing to turn in: no pairs
    trainer._set_usage(torch.eye(32, dtype=torch.float64))
    assert trainer.sample_batch().pair_first is None


def test_usage_matrix_separates_read_and_ignored_directions():
    trainer = T.Trainer(small_cfg(**{**WIDE, "weak_probe_codes": 12}, weak_pairs=2))
    with torch.no_grad():
        trainer.model.mapping[0].weight[:, 3:] = 0.0                # the decoder now ignores 29 coordinates
    usage = trainer.measure_usage()
    assert torch.allclose(usage, usage.T)
    lam, vec = torch.linalg.eigh(usage)
    lam, vec = lam.flip(0), vec.flip(1)
    assert float(lam[-1]) > -1e-12
    assert float(vec[:3, :3].square().sum()) > 2.7                  # the top three directions are the read ones
    # a floor between the read and the ignored directions; partners must then decode (almost) alike
    trainer.cfg = dataclasses.replace(trainer.cfg, weak_floor=float((lam[2] * lam[3]).sqrt().sqrt()))
    trainer._set_usage(usage)
    assert trainer._weak_basis.shape[1] == 29
    batch = trainer.sample_batch()
    base, partner = batch.codes[batch.pair_first], batch.codes[batch.pair_second]
    other = torch.nn.functional.normalize(base + 0.25 * torch.eye(32)[:2], dim=-1)   # a step along read axes
    x = batch.x.expand(2, -1, -1)
    with torch.no_grad():
        same = (trainer.model(base, x) - trainer.model(partner, x)).abs().mean()
        moved = (trainer.model(base, x) - trainer.model(other, x)).abs().mean()
    assert float(same) < 0.15 * float(moved)


def test_weak_pairs_train_and_stay_out_of_the_neighbour_search(monkeypatch):
    trainer = T.Trainer(small_cfg(**WIDE, weak_pairs=1, weak_refresh_every=2, weak_floor=1e9))
    seen = {"main": [], "pool": [], "pairs": [], "measured": 0}
    real_pool, real_pair, real_measure = T.L.nearest_pool, T.L.pair_push, trainer.measure_usage

    def pool(fb, pool_f, index, cfg_, **kwargs):
        seen["main"].append(fb.codes.shape[0])
        seen["pool"].append(pool_f.shape[0])
        return real_pool(fb, pool_f, index, cfg_, **kwargs)

    def pair(fb, first, second, chord_length, cfg_):
        seen["pairs"].append((fb.codes.shape[0], first.tolist(), second.tolist()))
        return real_pair(fb, first, second, chord_length, cfg_)

    def measure():
        seen["measured"] += 1
        return real_measure()

    monkeypatch.setattr(T.L, "nearest_pool", pool)
    monkeypatch.setattr(T.L, "pair_push", pair)
    trainer.measure_usage = measure
    before = [p.detach().clone() for p in trainer.model.parameters()]
    record = trainer.step()
    assert_finite(record)
    assert seen["main"] == [7, 7] and seen["pool"] == [7, 7]        # 8 codes a pass, one of them a partner
    assert seen["pairs"] == [(8, [0], [7]), (8, [0], [7])]
    assert record["weak/pair_rate"] > 0 and record["weak/used_directions"] == 0.0
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))
    assert trainer._main_rows is None
    trainer.step()
    trainer.step()
    assert seen["measured"] == 2                                     # at updates 0 and 2


def test_checkpoint_resume_with_weak_pairs_reproduces_the_next_steps(tmp_path):
    trainer = T.Trainer(small_cfg(**WIDE, weak_pairs=1, weak_refresh_every=4, weak_floor=1e9))
    for _ in range(3):
        trainer.step()
    path = trainer.save_checkpoint(str(tmp_path / "ckpt.pt"))
    expected = [trainer.step() for _ in range(3)]
    resumed = T.Trainer.from_checkpoint(path)
    assert torch.equal(resumed._usage, torch.load(path, weights_only=False)["usage"])
    got = [resumed.step() for _ in range(3)]
    for a, b in zip(expected, got):
        for key, value in a.items():
            if key.startswith("time_ms/") or key in ("gpu_peak_mb", "graph/build_ms") or isinstance(value, str):
                continue
            assert b[key] == pytest.approx(value, rel=1e-5, abs=1e-8), key


def test_frame_partners_follow_used_directions_and_turn_in_the_unused_ones():
    trainer = T.Trainer(small_cfg(**WIDE, weak_frame_passes=1, weak_frame_new=2, weak_chord=0.25))
    usage = torch.zeros(32, 32, dtype=torch.float64)
    usage[0, 0], usage[1, 1], usage[2, 2] = 3.0, 2.0, 1.0           # three used coordinates, strongest first
    trainer._set_usage(usage)
    plain = trainer.sample_batch()
    assert plain.pair_first is None and plain.frame_used == -1       # only a frame pass gets partners
    batch = trainer.sample_batch(frame=True)
    assert batch.codes.shape == (8, 32) and batch.frame_used == 3
    assert batch.pair_first.tolist() == [0] * 5 and batch.pair_second.tolist() == [3, 4, 5, 6, 7]
    centre, partner = batch.codes[0], batch.codes[3:]
    assert torch.allclose(partner.norm(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.allclose((partner - centre).norm(dim=-1), torch.full((5,), 0.25), atol=1e-5)
    assert torch.allclose(batch.pair_chord, torch.full((5,), 0.25), atol=1e-5)
    step = partner - centre
    for k in range(3):                                               # used partners: mostly along their own axis
        assert step[k, k].abs() > 0.8 * 0.25
    assert torch.allclose(partner[3:, :3], centre[:3].expand(2, -1), atol=1e-6)      # new partners keep the used part
    turns = torch.nn.functional.normalize(step[3:, 3:], dim=-1)
    assert abs(float((turns[0] * turns[1]).sum())) < 0.05           # and turn in perpendicular directions


def test_frame_pass_trains_and_only_the_first_pass_carries_a_frame(monkeypatch):
    trainer = T.Trainer(small_cfg(**WIDE, weak_frame_passes=1, weak_frame_new=2, weak_floor=1e9))
    seen = {"main": [], "pool": [], "frames": []}
    real_pool, real_frame = T.L.nearest_pool, T.L.frame_volume

    def pool(fb, pool_f, index, cfg_, **kwargs):
        seen["main"].append(fb.codes.shape[0])
        seen["pool"].append(pool_f.shape[0])
        return real_pool(fb, pool_f, index, cfg_, **kwargs)

    def frame(fb, centre, partners, chord_length, used, cfg_):
        seen["frames"].append((centre, partners.tolist(), used))
        return real_frame(fb, centre, partners, chord_length, used, cfg_)

    monkeypatch.setattr(T.L, "nearest_pool", pool)
    monkeypatch.setattr(T.L, "frame_volume", frame)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    record = trainer.step()
    assert_finite(record)
    assert seen["frames"] == [(0, [6, 7], 0)]                       # nothing counts as used above this floor
    assert seen["main"] == [6, 8] and seen["pool"] == [8, 6]        # the two partners stay out of the search
    assert record["weak/new_rms"] > 0 and "weak/frame_term" in record
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))
    with pytest.raises(ValueError):
        T.Trainer(small_cfg(**WIDE, weak_frame_passes=1, weak_pairs=1))


def test_anchor_file_posed_under_another_rule_is_refused(tmp_path):
    path = _anchor_file(tmp_path)                                     # no field: posed by mass (power 2)
    with pytest.raises(ValueError, match="pose_power"):
        T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2, pose_power=6.0))
    assert_finite(T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2)).step())
    stamped = _anchor_file(tmp_path, pose_power=6.0)
    with pytest.raises(ValueError, match="pose_power"):
        T.Trainer(small_cfg(**WIDE, anchor_path=stamped, anchor_codes_per_pass=2))
    assert_finite(T.Trainer(small_cfg(**WIDE, anchor_path=stamped, anchor_codes_per_pass=2, pose_power=6.0)).step())


def test_anchor_rows_can_be_left_out_of_the_size_term(tmp_path, monkeypatch):
    path = _anchor_file(tmp_path)
    rows = []
    real = T.L.pose_terms

    def pose_terms(fb, cfg_):
        rows.append(fb.f.shape[0])
        return real(fb, cfg_)

    monkeypatch.setattr(T.L, "pose_terms", pose_terms)
    trainer = T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2, anchor_exempt_size=True))
    assert_finite(trainer.step())
    full, main = max(rows), min(rows)
    assert main == full - 2                                           # the size term saw the batch without its 2 anchor rows
    rows.clear()
    assert_finite(T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2)).step())
    assert len(set(rows)) == 1                                        # off by default: every row counts


def _anchor_file(tmp_path, count=4, resolution=17, pose_power=None):
    """A small anchor file: ellipsoid-like fields, positive inside, in the stored layout."""
    axis = torch.linspace(-1.0, 1.0, resolution)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    grids = []
    for k in range(count):
        a, b, c = 1.0 + 0.2 * k, 1.0, 1.0 - 0.1 * k
        grids.append(0.5 - torch.sqrt((gx / a).square() + (gy / b).square() + (gz / c).square() + 1e-12))
    path = str(tmp_path / "anchors.pt")
    data = {"grid": torch.stack(grids).half(), "resolution": resolution, "positive_inside": True,
            "file_ids": list(range(count))}
    if pose_power is not None:
        data["pose_power"] = pose_power
    torch.save(data, path)
    return path


def test_anchor_codes_start_perpendicular_then_opposite():
    gen = torch.Generator().manual_seed(0)
    codes = T.anchor_code_init(6, 4, gen)
    assert torch.allclose(codes[:4] @ codes[:4].T, torch.eye(4), atol=1e-5)
    assert torch.allclose(codes[4:], -codes[:2])
    with pytest.raises(ValueError):
        T.anchor_code_init(9, 4, gen)


def test_anchors_take_the_last_rows_and_stay_out_of_the_neighbour_search(tmp_path, monkeypatch):
    path = _anchor_file(tmp_path)
    trainer = T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2))
    assert trainer.anchor_codes.shape == (4, 32) and trainer.anchor_grid.shape == (4, 17, 17, 17)
    start = trainer.anchor_codes.detach().clone()
    seen = {"main": [], "anchors": [], "codes": []}
    real_pool, real_match, real_field = T.L.nearest_pool, T.L.anchor_match, T.evaluate_field

    def pool(fb, pool_f, index, cfg_, **kwargs):
        seen["main"].append((fb.codes.shape[0], pool_f.shape[0]))
        return real_pool(fb, pool_f, index, cfg_, **kwargs)

    def match(fb, rows, grids, cfg_, **kwargs):
        seen["anchors"].append((rows.tolist(), tuple(grids.shape)))
        return real_match(fb, rows, grids, cfg_, **kwargs)

    def field(model, codes, *args, **kwargs):
        seen["codes"].append(codes.detach().clone())
        return real_field(model, codes, *args, **kwargs)

    monkeypatch.setattr(T.L, "nearest_pool", pool)
    monkeypatch.setattr(T.L, "anchor_match", match)
    monkeypatch.setattr(T, "evaluate_field", field)
    record = trainer.step()
    assert_finite(record)
    assert seen["main"] == [(6, 6), (6, 6)]                          # 8 codes a pass, 2 of them anchors
    assert seen["anchors"] == [([6, 7], (2, 17, 17, 17))] * 2
    used = torch.cat([codes[6:] for codes in seen["codes"]])         # the four anchors, each once per update
    assert torch.allclose(used @ start.T, torch.eye(4)[torch.argmax(used @ start.T, dim=1)], atol=1e-5)
    assert sorted(torch.argmax(used @ start.T, dim=1).tolist()) == [0, 1, 2, 3]
    assert record["anchor/distance"] > 0 and 0 <= record["anchor/iou"] <= 1 and record["anchor/lr"] == 0.0
    assert torch.equal(trainer.anchor_codes.detach(), start)         # fixed codes do not move
    with pytest.raises(ValueError):
        T.Trainer(small_cfg(**WIDE, anchor_path=path, weak_pairs=1))
    with pytest.raises(ValueError):
        T.Trainer(small_cfg(anchor_path=path))


def test_anchor_codes_are_frozen_first_then_move_and_stay_unit(tmp_path):
    path = _anchor_file(tmp_path)
    trainer = T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2, anchor_embed_lr=0.05,
                                  anchor_freeze_steps=2, anchor_ramp_steps=4))
    start = trainer.anchor_codes.detach().clone()
    assert [trainer._anchor_lr(s) for s in (0, 1, 2, 3, 5, 9)] == pytest.approx([0, 0, 0.0125, 0.025, 0.05, 0.05])
    trainer.step()
    trainer.step()
    assert torch.equal(trainer.anchor_codes.detach(), start)         # frozen for the first two updates
    record = trainer.step()
    assert record["anchor/lr"] == pytest.approx(0.0125)
    moved = trainer.anchor_codes.detach()
    assert not torch.equal(moved, start)
    assert torch.allclose(moved.norm(dim=-1), torch.ones(4), atol=1e-6)


def test_anchor_fit_improves_and_diagnostics_report_it(tmp_path):
    path = _anchor_file(tmp_path)
    trainer = T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2, lr=2e-3))
    first = trainer.run_diagnostics()
    for key in ("anchor/distance_mean", "anchor/iou_mean", "anchor/iou_min", "anchor/code_cos_max"):
        assert math.isfinite(first[key])
    assert abs(first["anchor/code_cos_max"]) < 1e-5                  # four perpendicular codes
    for _ in range(40):
        trainer.step()
    later = trainer.run_diagnostics()
    assert later["anchor/distance_mean"] < 0.8 * first["anchor/distance_mean"]


def test_checkpoint_resume_with_anchors_reproduces_the_next_steps(tmp_path):
    path = _anchor_file(tmp_path)
    trainer = T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2, anchor_embed_lr=0.02,
                                  anchor_freeze_steps=1, anchor_ramp_steps=2))
    for _ in range(3):
        trainer.step()
    ckpt = trainer.save_checkpoint(str(tmp_path / "ckpt.pt"))
    expected = [trainer.step() for _ in range(3)]
    resumed = T.Trainer.from_checkpoint(ckpt)
    got = [resumed.step() for _ in range(3)]
    for a, b in zip(expected, got):
        for key, value in a.items():
            if key.startswith("time_ms/") or key in ("gpu_peak_mb", "graph/build_ms") or isinstance(value, str):
                continue
            assert b[key] == pytest.approx(value, rel=1e-5, abs=1e-8), key
    assert torch.allclose(resumed.anchor_codes, trainer.anchor_codes, atol=1e-6)


def test_full_frame_fills_the_pass_and_needs_no_usage_measurement():
    trainer = T.Trainer(small_cfg(**WIDE, weak_frame_passes=1, weak_frame_full=True, weak_chord=0.25))
    batch = trainer.sample_batch(frame=True)
    assert batch.codes.shape == (8, 32) and batch.frame_used == 7     # 8 codes a pass: the centre and 7 partners
    centre, partner = batch.codes[0], batch.codes[1:]
    assert batch.pair_second.tolist() == list(range(1, 8)) and batch.pair_first.tolist() == [0] * 7
    assert torch.allclose((partner - centre).norm(dim=-1), torch.full((7,), 0.25), atol=1e-5)
    steps = torch.nn.functional.normalize(partner - (partner @ centre)[:, None] * centre, dim=-1)
    assert torch.allclose(steps @ steps.T, torch.eye(7), atol=1e-4)   # perpendicular directions
    record = trainer.step()
    assert_finite(record)
    assert trainer._usage is None and "weak/usage_effective" not in record
    assert 1.0 <= record["weak/frame_effective"] <= 7.0 and record["weak/frame_rms_top"] > 0


def test_a_frame_pass_and_anchors_share_an_update(tmp_path, monkeypatch):
    path = _anchor_file(tmp_path)
    trainer = T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2, weak_frame_passes=1,
                                  weak_frame_full=True))
    seen = {"main": [], "frames": 0, "anchors": []}
    real_pool, real_frame, real_match = T.L.nearest_pool, T.L.frame_volume, T.L.anchor_match

    def pool(fb, pool_f, index, cfg_, **kwargs):
        seen["main"].append((fb.codes.shape[0], pool_f.shape[0]))
        return real_pool(fb, pool_f, index, cfg_, **kwargs)

    def frame(*args, **kwargs):
        seen["frames"] += 1
        return real_frame(*args, **kwargs)

    def match(fb, rows, grids, cfg_, **kwargs):
        seen["anchors"].append(rows.tolist())
        return real_match(fb, rows, grids, cfg_, **kwargs)

    monkeypatch.setattr(T.L, "nearest_pool", pool)
    monkeypatch.setattr(T.L, "frame_volume", frame)
    monkeypatch.setattr(T.L, "anchor_match", match)
    record = trainer.step()
    assert_finite(record)
    assert seen["frames"] == 1 and seen["anchors"] == [[6, 7]]       # pass 0 the frame, pass 1 the anchors
    assert seen["main"] == [(1, 6), (6, 1)]
    assert "weak/frame_term" in record and "anchor/distance" in record


def test_anchor_neighbourhood_meets_the_anchor_nearby_with_a_fading_pull(tmp_path):
    path = _anchor_file(tmp_path)
    trainer = T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2, anchor_radius=0.3))
    table = torch.nn.functional.normalize(trainer.anchor_codes.detach(), dim=-1)
    distances, pulls = [], []
    for _ in range(40):
        index = torch.tensor([0, 3])
        batch = trainer.sample_batch(anchors=index)
        met = batch.codes[-2:]
        assert torch.allclose(met.norm(dim=-1), torch.ones(2), atol=1e-5)
        away = (met - table[index]).norm(dim=-1)
        assert torch.allclose(batch.anchor_pull, torch.exp(-(away / 0.3).square()), atol=1e-4)
        distances.append(away)
        pulls.append(batch.anchor_pull)
    distances, pulls = torch.cat(distances), torch.cat(pulls)
    assert float(distances.max()) <= 0.6 + 1e-5 and float(distances.min()) < 0.1 and float(distances.mean()) > 0.2
    assert float(pulls.min()) > 0.018 and float(pulls.max()) > 0.9
    record = trainer.step()
    assert_finite(record)
    plain = T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2))
    assert plain.sample_batch(anchors=torch.tensor([0, 3])).anchor_pull is None


def test_anchor_rows_can_be_left_out_of_the_field_accuracy_constraints(tmp_path, monkeypatch):
    path = _anchor_file(tmp_path)
    seen = {}
    real_consistency, real_eikonal, real_barrier = T.L.consistency, T.L.eikonal, T.L.barrier

    def counted(name, real):
        def inner(fb, *args):
            seen.setdefault(name, []).append(fb.codes.shape[0])
            return real(fb, *args)
        return inner

    monkeypatch.setattr(T.L, "consistency", counted("consistency", real_consistency))
    monkeypatch.setattr(T.L, "eikonal", counted("eikonal", real_eikonal))
    monkeypatch.setattr(T.L, "barrier", counted("barrier", real_barrier))
    trainer = T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2, anchor_exempt_field=True))
    assert_finite(trainer.step())
    assert seen["consistency"] == [6, 6] and seen["eikonal"] == [6, 6]     # 8 codes a pass, 2 anchors left out
    assert seen["barrier"] == [8, 8]                                       # every other term still sees them
    seen.clear()
    plain = T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2))
    plain.step()
    assert seen["consistency"] == [8, 8]


def test_consistency_tolerance_schedule_holds_then_tightens():
    trainer = T.Trainer(small_cfg(**WIDE, consistency_tolerance_start=0.03, consistency_hold_steps=2,
                                  consistency_tighten_steps=4))
    final = trainer.cfg.tolerances["consistency"]
    values = [trainer._consistency_tolerance(s) for s in (0, 1, 2, 4, 6, 50)]
    assert values == pytest.approx([0.03, 0.03, 0.03, math.sqrt(0.03 * final), final, final])
    record = trainer.step()
    assert record["tol/consistency"] == pytest.approx(0.03)
    for _ in range(6):
        record = trainer.step()
    assert record["tol/consistency"] == pytest.approx(final)
    assert T.Trainer(small_cfg(**WIDE))._consistency_tolerance(0) is None


def test_field_moves_surface_reaches_the_neighbour_term_and_trains(tmp_path, monkeypatch):
    path = _anchor_file(tmp_path)
    trainer = T.Trainer(small_cfg(**WIDE, anchor_path=path, anchor_codes_per_pass=2, field_moves_surface=True,
                                  consistency_fixed_surface=True, consistency_min_depth=0.04))
    seen = []
    real_pool = T.L.nearest_pool

    def pool(fb, pool_f, index, cfg_, **kwargs):
        seen.append(fb.f_move is not None and fb.f_move.shape == fb.f.shape and fb.f_move.requires_grad)
        return real_pool(fb, pool_f, index, cfg_, **kwargs)

    monkeypatch.setattr(T.L, "nearest_pool", pool)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    for _ in range(2):
        assert_finite(trainer.step())
    assert seen and all(seen)
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))


def test_tidy_stage_holds_the_shapes_against_a_frozen_copy():
    trainer = T.Trainer(small_cfg(**WIDE, consistency_fixed_surface=True, consistency_min_depth=0.04,
                                  variety_weight=0.0, shape_hold_weight=100.0, shape_hold_band=0.05))
    start = [p.detach().clone() for p in trainer.model.parameters()]
    first = trainer.step()
    assert_finite(first)
    assert first["hold/moved"] == 0.0                              # the copy is taken at the first update
    assert all(torch.equal(a, b) for a, b in zip(start, trainer._reference.parameters()))
    later = [trainer.step() for _ in range(3)][-1]
    assert_finite(later)
    assert 0.0 < later["hold/moved"] < 0.01                        # the field near the old surfaces barely moves
    assert all(torch.equal(a, b) for a, b in zip(start, trainer._reference.parameters()))   # the copy stays frozen
    assert any(not torch.equal(a, b) for a, b in zip(start, trainer.model.parameters()))


def test_code_isotropy_is_zero_for_an_even_side_network_and_only_pushes_the_side_network():
    cfg = small_cfg(**WIDE, mapping_layers=0)
    trainer = T.Trainer(cfg)
    model = trainer.model
    gen = torch.Generator().manual_seed(3)
    codes = torch.nn.functional.normalize(torch.randn(5, cfg.latent_dim, generator=gen), dim=-1).to(trainer.device)
    with torch.no_grad():                                     # orthonormal columns: every direction counts the same
        q, _ = torch.linalg.qr(torch.randn(model.film.weight.shape, generator=gen))
        model.film.weight.copy_(q.to(model.film.weight))
    term, ratio = T.L.code_isotropy(model, codes, 8, gen)
    assert float(term) < 1e-8 and float(ratio) == pytest.approx(1.0, abs=1e-3)
    with torch.no_grad():                                     # one code direction made ten times stronger
        model.film.weight[:, 0] *= 10.0
    term, ratio = T.L.code_isotropy(model, codes, 8, gen)
    assert float(term) > 0.05 and float(ratio) > 1.2
    term.backward()
    assert model.film.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in list(model.hidden.parameters()) + list(model.out.parameters()))


def test_code_isotropy_term_trains_and_is_off_by_default():
    assert "iso/term" not in T.Trainer(small_cfg(**WIDE)).step()
    trainer = T.Trainer(small_cfg(**WIDE, code_isotropy_weight=1.0, code_isotropy_directions=6))
    first = trainer.step()
    assert_finite(first)
    assert first["iso/term"] >= 0 and first["iso/ratio"] >= 1.0
    for _ in range(3):
        assert_finite(trainer.step())


def test_field_ramp_raises_the_limits_of_the_two_field_weights():
    trainer = T.Trainer(small_cfg(**WIDE, field_ramp_steps=4, field_ramp_start=0.01,
                                  mult_ranges={"consistency": (1.0, 100.0)}))
    assert trainer._field_limits(0) == pytest.approx({"eikonal": 0.01, "consistency": 0.01})
    assert trainer._field_limits(2)["consistency"] == pytest.approx(1.0)          # geometric: halfway is the root
    assert trainer._field_limits(4) == pytest.approx({"eikonal": trainer.cfg.mult_max, "consistency": 100.0})
    seen = []
    for _ in range(6):
        record = trainer.step()
        assert_finite(record)
        seen.append((record["mult/consistency"], record["mult/eikonal"]))
    assert seen[0][0] <= 0.011 and seen[0][1] <= 0.011                            # next to nothing at first
    assert all(c <= 100.0 and e <= trainer.cfg.mult_max for c, e in seen)
    assert trainer.multipliers.range("consistency") == (1.0, 100.0)               # back at the configured range
    assert T.Trainer(small_cfg(**WIDE))._field_limits(0) == {}                    # off by default


def test_tidy_stage_can_pin_the_surface_itself_with_newton_steps():
    trainer = T.Trainer(small_cfg(**WIDE, consistency_fixed_surface=True, consistency_min_depth=0.04,
                                  variety_weight=0.0, shape_hold_weight=100.0, shape_hold_newton=4))
    start = [p.detach().clone() for p in trainer.model.parameters()]
    first = trainer.step()
    assert_finite(first)
    assert first["hold/arrived"] > 0.5                              # most points reach the copy's surface
    assert first["hold/moved"] < 2.1e-3                             # and the decoder is still zero there
    later = [trainer.step() for _ in range(3)][-1]
    assert_finite(later)
    assert later["hold/moved"] < 0.01
    assert all(torch.equal(a, b) for a, b in zip(start, trainer._reference.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(start, trainer.model.parameters()))


def test_field_only_drops_every_push_but_the_two_field_terms(monkeypatch):
    trainer = T.Trainer(small_cfg(**WIDE, variety_weight=0.0, field_only=True))
    seen = []
    real = trainer.multipliers.combine

    def combine(objective, constraints):
        seen.append(sorted(constraints))
        return real(objective, constraints)

    monkeypatch.setattr(trainer.multipliers, "combine", combine)
    record = trainer.step()
    assert_finite(record)
    assert seen and all(names == ["consistency", "eikonal"] for names in seen)
    assert "term/pose_scale" in record                              # still measured and logged


def test_eikonal_leaves_out_the_points_the_consistency_term_leaves_out():
    trainer = T.Trainer(small_cfg(**WIDE, consistency_fixed_surface=True, consistency_min_depth=0.05))
    batch = trainer.sample_batch()
    fb = T.evaluate_field(trainer.model, batch.codes, batch.cluster, batch.x, trainer.cfg)
    error = (fb.g[:, 1:].norm(dim=-1) - 1.0).square()
    weight = fb.fp_weight[:, 1:]
    assert 0 < float(weight.mean()) < 1
    assert float(T.L.eikonal(fb)) == pytest.approx(float((error * weight).sum() / weight.sum()), rel=1e-4)
    assert float(T.L.eikonal(fb)) != pytest.approx(float(error.mean()), rel=1e-4)


def test_fixed_surface_consistency_trains():
    trainer = T.Trainer(small_cfg(**WIDE, consistency_fixed_surface=True, consistency_min_depth=0.04))
    before = [p.detach().clone() for p in trainer.model.parameters()]
    for _ in range(2):
        assert_finite(trainer.step())
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))


def test_checkpoint_resume_with_a_frame_pass_reproduces_the_next_steps(tmp_path):
    trainer = T.Trainer(small_cfg(**WIDE, weak_frame_passes=1, weak_frame_new=2, weak_refresh_every=4))
    for _ in range(3):
        trainer.step()
    path = trainer.save_checkpoint(str(tmp_path / "ckpt.pt"))
    expected = [trainer.step() for _ in range(3)]
    resumed = T.Trainer.from_checkpoint(path)
    got = [resumed.step() for _ in range(3)]
    for a, b in zip(expected, got):
        for key, value in a.items():
            if key.startswith("time_ms/") or key in ("gpu_peak_mb", "graph/build_ms") or isinstance(value, str):
                continue
            assert b[key] == pytest.approx(value, rel=1e-5, abs=1e-8), key
