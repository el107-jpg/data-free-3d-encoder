def sub(s, a, b, tag):
    assert a in s, tag
    return s.replace(a, b)


p = 'tests/test_data_free_core.py'
s = open(p).read()
s = sub(s, '''    "eikonal", "consistency", "barrier", "origin", "volume", "pose",
''', '''    "eikonal", "consistency", "barrier", "origin", "volume", "pose",
    "pose_centroid", "pose_scale", "pose_axes",
''', 1)
s = sub(s, '''    assert cfg.cov_trace_target == pytest.approx(3 * 0.5 ** 2 / 5)
''', '''    # trace of the R = 0.5 ball as the soft indicator (tau = 0.02) measures it
    assert cfg.cov_trace_target == pytest.approx(0.155527, abs=1e-6)
    assert DataFreeConfig(soft_tau=1e-6).cov_trace_target == pytest.approx(3 * 0.5 ** 2 / 5)
''', 2)
open(p, 'w').write(s)

p = 'tests/test_data_free_losses.py'
s = open(p).read()
s = sub(s, '''    assert cfg.cov_trace_target == pytest.approx(3 * R * R / 5)
''', '''    assert cfg.cov_trace_target == pytest.approx(L.soft_ball_trace(R, cfg.soft_tau))
''', 3)
s = sub(s, '''    # With the sampling noise gone, what is left is the bias of the soft inside
    # indicator: it inflates the covariance trace by about 14 pi^2 tau^2 / (6 R^2)
    # (3.7% at tau = 0.02, R = 0.5), i.e. a pose value near 3e-5.
    fb = make_fb(sphere(), box_points(1, 400_000, seed=3))
    value = float(L.pose(fb, cfg))
    print(f"pose on exact sphere, N=400000: {value:.3e}")
    assert value < 5e-5
    assert value < cfg.tolerances["pose"]
''', '''    # With the sampling noise gone nothing is left: the default trace target is
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
    assert biased > 5 * max(cfg.tolerances[k] for k in L.POSE_KEYS)


def test_pose_terms_report_the_true_squared_error(cfg):
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
''', 4)
open(p, 'w').write(s)

p = 'tests/test_data_free_train.py'
s = open(p).read()
s = sub(s, '''    cfg.tolerances["pose"] = 1e9           # always met
''', '''    cfg.tolerances["pose_scale"] = 1e9     # always met
''', 5)
s = sub(s, '''    assert record["mult/pose"] == pytest.approx(''', '''    assert record["mult/pose_scale"] == pytest.approx(''', 6)
s = sub(s, '''def test_loss_is_objective_plus_weighted_constraints():''',
        '''def test_pose_enters_as_three_separate_constraints():
    assert T.FIELD_KEYS[-3:] == ("pose_centroid", "pose_scale", "pose_axes")
    assert "pose" not in T.CONSTRAINT_KEYS
    cfg = small_cfg()
    for key in T.FIELD_KEYS[-3:]:
        assert key in cfg.tolerances
    record = T.Trainer(cfg).step()
    assert "term/pose" not in record and "mult/pose" not in record
    for key in T.FIELD_KEYS[-3:]:
        assert math.isfinite(record[f"term/{key}"]) and record[f"mult/{key}"] > 0


def test_loss_is_objective_plus_weighted_constraints():''', 7)
s = sub(s, '''                "descriptor_spread", "probe_field_distance", "e"):''',
        '''                "descriptor_spread", "probe_field_distance", "e",
                "graph_vs_grid/confirm_dropped_kind0", "graph_vs_grid/confirm_dropped_cavities",
                "graph_vs_grid/raw_cavities", "dead/subspace_stability_mean",
                "dead/subspace_stability_count", "pose/centroid_offset_max",
                "pose/trace_rel_error_max", "pose/offdiag_abs_max"):''', 8)
s = sub(s, '''    assert record["grid/clean_fraction"] == 1.0
''', '''    assert record["grid/clean_fraction"] == 1.0
    # ... centred and of the target size, as measured with many points
    assert record["pose/centroid_offset_max"] < 0.01
    assert record["pose/trace_rel_error_max"] < 0.03
''', 9)
s = sub(s, '''# ------------------------------------------------------------------------ CLI''',
        '''def test_diagnostics_measure_a_planted_pose_error():
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


# ------------------------------------------------------------------------ CLI''', 10)
open(p, 'w').write(s)

p = 'tests/test_data_free_diagnostics.py'
s = open(p).read()
s = sub(s, '''    rows = graph_vs_grid(decoder, codes, cfg, resolution=32, from_rows=True,
                         generator=torch.Generator().manual_seed(0))
    assert rows["both_agreement"] == 1.0
''', '''    rows = graph_vs_grid(decoder, codes, cfg, resolution=32, from_rows=True,
                         generator=torch.Generator().manual_seed(0))
    assert rows["both_agreement"] == 1.0
    assert out["confirm_dropped"] == {"kind0": 0, "kind1": 0, "cavities": 0}
    assert out["graph_counts_raw"]["pieces"].tolist() == [1, 1]


def test_graph_vs_grid_counts_after_confirm(monkeypatch):
    """``confirm`` is part of the method: what it drops must not count as a disagreement."""
    import dataclasses

    import arm_planner.data_free.graph as graph_module
    from arm_planner.data_free.diagnostics import graph_vs_grid

    cfg, decoder, codes = _real_decoder()
    real = graph_module.build_graph

    def with_false_cavity(fb, e, cfg_, mode="full", **kwargs):
        graph = real(fb, e, cfg_, mode=mode, **kwargs)
        device = graph.inside.device
        mask = torch.zeros(1, graph.inside.shape[1], dtype=torch.bool, device=device)
        mask[0, 5] = True
        cavities = graph.num_cavities.clone()
        cavities[0] += 1
        # joining segment far outside the shape: every sample is free space, so
        # the "cavity" is open to the far field and confirm drops it
        return dataclasses.replace(
            graph,
            cav_shape=torch.zeros(1, dtype=torch.long, device=device),
            cav_a=torch.tensor([[0.80, 0.0, 0.0]], device=device),
            cav_b=torch.tensor([[0.95, 0.0, 0.0]], device=device),
            cav_mask=mask, num_cavities=cavities,
        )

    monkeypatch.setattr(graph_module, "build_graph", with_false_cavity)
    gen = lambda: torch.Generator().manual_seed(0)  # noqa: E731
    raw = graph_vs_grid(decoder, codes, cfg, resolution=32, generator=gen(), confirmed=False)
    assert raw["graph_counts"]["cavities"].tolist() == [1, 0]
    assert raw["cavities_agreement"] == 0.5
    out = graph_vs_grid(decoder, codes, cfg, resolution=32, generator=gen())
    assert out["graph_counts_raw"]["cavities"].tolist() == [1, 0]
    assert out["graph_counts"]["cavities"].tolist() == [0, 0]
    assert out["confirm_dropped"]["cavities"] == 1
    assert out["both_agreement"] == 1.0
    rows = graph_vs_grid(decoder, codes, cfg, resolution=32, generator=gen(), from_rows=True)
    assert rows["graph_counts"]["cavities"].tolist() == [0, 0]
''', 11)
open(p, 'w').write(s)
print("tests patched")
