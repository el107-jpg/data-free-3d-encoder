"""Tests for arm_planner/data_free/diagnostics.py.

Run as a script to print the measured numbers:
    PYTHONPATH=. python tests/test_data_free_diagnostics.py
"""

import math
from types import SimpleNamespace

import pytest
import torch

from arm_planner.data_free.diagnostics import (
    compare_graph_to_grid,
    dead_directions,
    decoder_grid_topology,
    graph_topology,
    grid_topology,
    subspace_stability,
    summarise,
)

# ---------------------------------------------------------------------------
# Analytic fields, positive inside
# ---------------------------------------------------------------------------


def _sphere(centre, radius):
    return lambda p: radius - (p - torch.tensor(centre, dtype=p.dtype, device=p.device)).norm(dim=1)


def _torus(p, major=0.55, minor=0.2, centre=(0.0, 0.0, 0.0)):
    p = p - torch.tensor(centre, dtype=p.dtype, device=p.device)
    q = torch.stack([p[:, :2].norm(dim=1) - major, p[:, 2]], dim=1)
    return minor - q.norm(dim=1)


def _two_spheres(p):
    return torch.maximum(_sphere([0.45, 0.0, 0.0], 0.3)(p), _sphere([-0.45, 0.0, 0.0], 0.3)(p))


def _double_torus(p):
    return torch.maximum(
        _torus(p, 0.3, 0.12, (0.3, 0.0, 0.0)), _torus(p, 0.3, 0.12, (-0.3, 0.0, 0.0))
    )


def _box_two_holes(p):
    box = torch.minimum(
        torch.minimum(0.8 - p[:, 0].abs(), 0.5 - p[:, 1].abs()), 0.3 - p[:, 2].abs()
    )
    offset = torch.tensor([0.4, 0.0], dtype=p.dtype, device=p.device)
    hole_a = (p[:, :2] - offset).norm(dim=1) - 0.2
    hole_b = (p[:, :2] + offset).norm(dim=1) - 0.2
    return torch.minimum(box, torch.minimum(hole_a, hole_b))


def _hollow_shell(p):
    r = p.norm(dim=1)
    return torch.minimum(0.7 - r, r - 0.4)


# name -> (field, pieces, cavities, handles)
ANALYTIC = {
    "sphere": (_sphere([0.0, 0.0, 0.0], 0.5), 1, 0, 0),
    "two spheres": (_two_spheres, 2, 0, 0),
    "torus": (_torus, 1, 0, 1),
    "double torus": (_double_torus, 1, 0, 2),
    "box, two through-holes": (_box_two_holes, 1, 0, 2),
    "hollow shell": (_hollow_shell, 1, 1, 0),
}


@pytest.mark.parametrize("resolution", [48, 64])
@pytest.mark.parametrize("name", list(ANALYTIC))
def test_grid_topology_analytic(name, resolution):
    field, pieces, cavities, handles = ANALYTIC[name]
    out = grid_topology(field, resolution=resolution)
    assert (out["pieces"], out["cavities"], out["handles"]) == (pieces, cavities, handles)
    assert out["euler"] == pieces + cavities - handles
    assert 0.0 < out["volume_fraction"] < 1.0


def test_grid_topology_solid_touching_boundary_and_empty():
    # A slab across the whole box splits the free space in two; both halves
    # touch the boundary, so neither is a cavity.
    out = grid_topology(lambda p: 0.2 - p[:, 2].abs(), resolution=24)
    assert (out["pieces"], out["cavities"], out["handles"]) == (1, 0, 0)
    empty = grid_topology(lambda p: -torch.ones(p.shape[0]), resolution=16)
    assert (empty["pieces"], empty["cavities"], empty["handles"]) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Fake decoders with analytically known shape families
# ---------------------------------------------------------------------------

DIM = 32
RADIUS = 0.5


def _smooth_modes(u):
    """Degree-2 and degree-3 harmonics of the unit direction: orthogonal to
    translation (degree 1) and scale (degree 0) on a sphere."""
    x, y, z = u.unbind(-1)
    return [x * y, y * z, 3 * z * z - 1, x * y * z, z * (x * x - y * y)]


def _wobble_mode(u, omega=40.0):
    x, y, z = u.unbind(-1)
    return torch.sin(omega * x) * torch.sin(omega * y) * torch.sin(omega * z)


class FakeDecoder:
    """Same ``prepare`` / ``query`` interface as the real decoder.

    kind 'pose':   sphere translated by gain*z[0:3] and scaled by 1 + gain*z[3].
    kind 'smooth': R + gain * sum_k z[k] * Y_k(x/|x|) - |x|, k < modes.
    kind 'wobble': 'smooth' plus gain * z[modes] * high-frequency angular term.
    Code coordinates 8.. do nothing in every family.
    """

    def __init__(self, kind, modes=4, gain=0.3):
        self.kind, self.modes, self.gain = kind, modes, gain

    def prepare(self, codes):
        return codes / codes.norm(dim=-1, keepdim=True)

    def query(self, mod, x):
        z = mod[:, None, :]  # B×1×D
        if self.kind == "pose":
            centre = self.gain * z[..., 0:3]
            scale = 1.0 + self.gain * z[..., 3]
            return scale * RADIUS - torch.sqrt(((x - centre) ** 2).sum(-1) + 1e-12)
        rho = torch.sqrt((x**2).sum(-1) + 1e-12)
        u = x / rho[..., None]
        radius = torch.full_like(rho, RADIUS)
        for k, mode in enumerate(_smooth_modes(u)[: self.modes]):
            radius = radius + self.gain * z[..., k] * mode
        if self.kind == "wobble":
            radius = radius + self.gain * z[..., self.modes] * _wobble_mode(u)
        return radius - rho

    def forward(self, codes, x):
        return self.query(self.prepare(codes), x)


def _sphere_code(seed=0, zero=8):
    """A unit code whose first ``zero`` coordinates are 0: every fake family is
    then exactly the base sphere at this code."""
    z = torch.randn(DIM, generator=torch.Generator().manual_seed(seed))
    z[:zero] = 0.0
    return z / z.norm()


def _generic_code(seed=0):
    z = torch.randn(DIM, generator=torch.Generator().manual_seed(seed))
    return z / z.norm()


def _run(kind, code, **kwargs):
    gen = torch.Generator().manual_seed(1)
    return dead_directions(FakeDecoder(kind), code, None, generator=gen, **kwargs)


def test_dead_directions_pose_only_family():
    for code in (_sphere_code(), _generic_code()):
        out = _run("pose", code)
        assert out["surface_points"] > 2000
        assert out["live"] == 0
        assert out["pose_share"] > 0.999
        assert out["rms_motion"] > 1e-2  # the surface does move; it is all pose
        assert out["singular_values"].shape == (DIM - 1,)
        assert out["directions"].shape == (DIM - 1, DIM)


def test_dead_directions_smooth_family_counts_modes():
    code = _sphere_code()
    out = _run("smooth", code)
    assert out["live"] == 4
    assert out["wobble_share"] < 0.1
    assert out["pose_share"] < 0.05
    s = out["singular_values"]
    assert s[3] > 100 * s[4]
    # live directions lie in the span of the first four code coordinates and
    # are tangent to the sphere at the code
    live = out["directions"][:4]
    assert float((live[:, :4] ** 2).sum()) / 4 > 0.99
    assert float((live @ code).abs().max()) < 1e-4
    assert torch.allclose(live.norm(dim=1), torch.ones(4), atol=1e-4)


def test_dead_directions_generic_code_rotation_overlap():
    # Away from the base sphere a rotation of the deformed shape is itself a
    # low-order deformation, so pose removal may absorb part of the family:
    # never more than the 4 modes, never fewer than 4 - 3 rotations.
    out = _run("smooth", _generic_code())
    assert 1 <= out["live"] <= 4
    assert out["wobble_share"] < 0.15


def test_dead_directions_wobble_removed_by_smoothing():
    code = _sphere_code()
    smooth = _run("smooth", code)
    wobble = _run("wobble", code)
    assert wobble["wobble_share"] > smooth["wobble_share"] + 0.1
    # the four smooth modes are untouched by the extra direction
    assert torch.allclose(wobble["singular_values"][:3], smooth["singular_values"][:3], rtol=0.02)

    # the wobble direction alone: one code direction moves the surface, and
    # smoothing removes most of its energy
    gen = torch.Generator().manual_seed(1)
    alone = dead_directions(FakeDecoder("wobble", modes=0), code, None, generator=gen)
    raw, s = alone["singular_values_unsmoothed"], alone["singular_values"]
    assert raw[0] > 0.05 and raw[1] < 1e-3
    assert s[0] < 0.5 * raw[0]
    assert alone["wobble_share"] > 0.75
    assert abs(float(alone["directions"][0, 0])) > 0.99


def test_subspace_stability_and_summarise():
    a = _run("smooth", _sphere_code(0))
    b = _run("smooth", _sphere_code(1))
    assert subspace_stability(a, b) > 0.98
    assert subspace_stability(a, a) == pytest.approx(1.0, abs=1e-5)
    # an unrelated 4-D subspace overlaps only by chance
    other = dict(a)
    other["directions"] = torch.roll(a["directions"], shifts=16, dims=1)
    assert subspace_stability(a, other) < 0.2
    dead = _run("pose", _sphere_code())
    assert math.isnan(subspace_stability(a, dead))
    assert subspace_stability(a, dead, k=2) >= 0.0

    summary = summarise([a, b, dead])
    assert summary["probes"] == 3
    assert summary["live_mean"] == pytest.approx(8 / 3)
    assert summary["live_min"] == 0.0 and summary["live_max"] == 4.0
    assert 0.0 <= summary["wobble_share_mean"] <= 1.0
    assert 0.0 <= summary["pose_share_mean"] <= 1.0
    assert all(isinstance(v, (int, float)) for v in summary.values())
    assert summarise([]) == {"probes": 0}


def test_decoder_grid_topology_fake():
    codes = torch.stack([_sphere_code(0), _generic_code(1)])
    out = decoder_grid_topology(FakeDecoder("smooth"), codes, None, resolution=32)
    assert len(out) == 2
    for item in out:
        assert (item["pieces"], item["cavities"], item["handles"]) == (1, 0, 0)


# ---------------------------------------------------------------------------
# Graph versus grid
# ---------------------------------------------------------------------------


def test_graph_topology_from_spec_fields():
    graph = SimpleNamespace(
        main_label=torch.zeros(4, dtype=torch.long),
        frag_shape=torch.tensor([0, 0, 2, 2, 2]),
        frag_kind=torch.tensor([0, 1, 0, 0, 1]),
        cav_shape=torch.tensor([3, 3, 1]),
    )
    counts = graph_topology(graph)
    assert counts["pieces"].tolist() == [2, 1, 3, 1]
    assert counts["cavities"].tolist() == [0, 1, 0, 2]

    empty = SimpleNamespace(
        main_label=torch.zeros(3, dtype=torch.long),
        frag_shape=torch.zeros(0, dtype=torch.long),
        frag_kind=torch.zeros(0, dtype=torch.long),
        cav_shape=torch.zeros(0, dtype=torch.long),
    )
    counts = graph_topology(empty)
    assert counts["pieces"].tolist() == [1, 1, 1]
    assert counts["cavities"].tolist() == [0, 0, 0]


def test_compare_graph_to_grid():
    graph_counts = {"pieces": torch.tensor([1, 2, 1, 3]), "cavities": torch.tensor([0, 0, 1, 0])}
    grid_counts = [
        {"pieces": 1, "cavities": 0},
        {"pieces": 1, "cavities": 0},
        {"pieces": 1, "cavities": 1},
        {"pieces": 4, "cavities": 1},
    ]
    out = compare_graph_to_grid(graph_counts, grid_counts)
    assert out["count"] == 4
    assert out["pieces_agreement"] == 0.5
    assert out["pieces_graph_more"] == 0.25 and out["pieces_graph_fewer"] == 0.25
    assert out["cavities_agreement"] == 0.75
    assert out["cavities_graph_fewer"] == 0.25 and out["cavities_graph_more"] == 0.0
    assert out["both_agreement"] == 0.5
    same = compare_graph_to_grid(graph_counts, graph_counts)
    assert same["both_agreement"] == 1.0
    with pytest.raises(ValueError):
        compare_graph_to_grid(graph_counts, grid_counts[:2])


# ---------------------------------------------------------------------------
# Real modules, when the other tasks' files are present
# ---------------------------------------------------------------------------


def _real_decoder():
    config = pytest.importorskip("arm_planner.data_free.config")
    decoder_mod = pytest.importorskip("arm_planner.data_free.decoder")
    latent = pytest.importorskip("arm_planner.data_free.latent")
    cfg = config.DataFreeConfig()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = decoder_mod.ModulatedSphereDecoder(cfg).to(device)
    codes = latent.sample_sphere(2, cfg.latent_dim, generator=torch.Generator().manual_seed(0)).to(device)
    return cfg, decoder, codes


def test_real_decoder_at_init_is_a_dead_sphere():
    cfg, decoder, codes = _real_decoder()
    out = dead_directions(decoder, codes[0], cfg, points=2048, generator=torch.Generator().manual_seed(0))
    assert out["surface_points"] > 1000
    assert out["live"] == 0
    topo = decoder_grid_topology(decoder, codes[:1], cfg, resolution=32)[0]
    assert (topo["pieces"], topo["cavities"], topo["handles"]) == (1, 0, 0)


def test_real_graph_matches_grid_at_init():
    cfg, decoder, codes = _real_decoder()
    pytest.importorskip("arm_planner.data_free.graph")
    from arm_planner.data_free.diagnostics import graph_vs_grid

    out = graph_vs_grid(decoder, codes, cfg, resolution=32, generator=torch.Generator().manual_seed(0))
    assert out["count"] == codes.shape[0]
    assert out["graph_counts"]["pieces"].tolist() == [1, 1]
    assert out["graph_counts"]["cavities"].tolist() == [0, 0]
    assert out["both_agreement"] == 1.0
    rows = graph_vs_grid(decoder, codes, cfg, resolution=32, from_rows=True,
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


def _analytic_graph_counts(names, points=2048, seed=0, e=1e-3):
    """Real ``build_graph`` on analytic fields (tuple input), against the grid."""
    from arm_planner.data_free.config import DataFreeConfig
    from arm_planner.data_free.graph import build_graph

    cfg = DataFreeConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(points, 3, generator=torch.Generator().manual_seed(seed)) * 2 - 1
    x[0] = 0.0
    x = x.to(device)
    packs = []
    for name in names:
        fn = ANALYTIC[name][0]
        xr = x.clone().requires_grad_(True)
        f = fn(xr)
        (g,) = torch.autograd.grad(f.sum(), xr)
        xp = (x - f[:, None] * g).detach()
        packs.append((x, f.detach(), g, xp, fn(xp).detach()))
    batch = tuple(torch.stack(t) for t in zip(*packs))
    graph = build_graph(batch, e, cfg, mode="full")
    grid = [grid_topology(ANALYTIC[name][0], resolution=48) for name in names]
    return graph, grid


def test_real_graph_on_analytic_sphere():
    pytest.importorskip("arm_planner.data_free.graph")
    graph, grid = _analytic_graph_counts(["sphere"])
    for from_rows in (False, True):
        out = compare_graph_to_grid(graph_topology(graph, from_rows=from_rows), grid)
        assert out["both_agreement"] == 1.0


# ---------------------------------------------------------------------------
# Measured numbers
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("grid topology: name, resolution -> pieces cavities handles euler")
    for name, (fn, *_expected) in ANALYTIC.items():
        for res in (48, 64):
            t0 = time.time()
            o = grid_topology(fn, resolution=res)
            print(f"  {name:24s} {res}: {o['pieces']} {o['cavities']} {o['handles']} {o['euler']}"
                  f"  ({time.time() - t0:.2f}s)")

    def show(label, o, n=6):
        sv = ", ".join(f"{v:.4f}" for v in o["singular_values"][:n].tolist())
        raw = ", ".join(f"{v:.4f}" for v in o["singular_values_unsmoothed"][:n].tolist())
        print(f"  {label}: live={o['live']} pose_share={o['pose_share']:.4f} "
              f"wobble_share={o['wobble_share']:.4f} e={o['consistency_error']:.2e} "
              f"thr={o['threshold']:.2e} rms={o['rms_motion']:.4f} S={o['surface_points']}")
        print(f"      s[:{n}]   = {sv}\n      raw[:{n}] = {raw}")

    print("dead directions, fake families")
    for kind in ("pose", "smooth", "wobble"):
        for label, code in (("sphere code", _sphere_code()), ("generic code", _generic_code())):
            t0 = time.time()
            o = _run(kind, code)
            show(f"{kind:6s} {label:12s} ({time.time() - t0:.2f}s)", o)
    for pts in (4096, 16384):
        o = dead_directions(FakeDecoder("wobble", modes=0), _sphere_code(), None, points=pts,
                            generator=torch.Generator().manual_seed(1))
        show(f"wobble alone, points={pts}", o, n=2)
    try:
        cfg, decoder, codes = _real_decoder()
    except BaseException as error:  # other tasks' modules not present
        print("real decoder not available:", error)
    else:
        print("real decoder on", codes.device)
        for label in ("at init", "all parameters + 0.02*noise (not a trained model)"):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            o = dead_directions(decoder, codes[0], cfg, generator=torch.Generator().manual_seed(0))
            dt = time.time() - t0
            show(f"{label} ({dt:.2f}s)", o)
            t0 = time.time()
            topo = decoder_grid_topology(decoder, codes, cfg, resolution=64)
            print(f"      grid topology 64, {len(topo)} codes ({time.time() - t0:.2f}s):",
                  [(t["pieces"], t["cavities"], t["handles"]) for t in topo])
            if torch.cuda.is_available():
                print(f"      peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
            with torch.no_grad():
                for prm in decoder.parameters():
                    prm.add_(0.02 * torch.randn_like(prm))
    try:
        from arm_planner.data_free.diagnostics import graph_vs_grid
        import arm_planner.data_free.graph  # noqa: F401
    except ImportError as error:
        print("real graph not available:", error)
    else:
        print("real graph vs grid (48) on analytic fields, e=1e-3: name -> graph (pieces, cavities) | "
              "rows | grid")
        names = ["sphere", "two spheres", "torus", "double torus", "hollow shell"]
        for seed in (0, 1, 2):
            graph, grid = _analytic_graph_counts(names, seed=seed)
            c, r = graph_topology(graph), graph_topology(graph, from_rows=True)
            for i, name in enumerate(names):
                print(f"  seed {seed} {name:14s}: ({int(c['pieces'][i])}, {int(c['cavities'][i])}) | "
                      f"({int(r['pieces'][i])}, {int(r['cavities'][i])}) | "
                      f"({grid[i]['pieces']}, {grid[i]['cavities']})")
            print("   ", {k: v for k, v in compare_graph_to_grid(c, grid).items()})
        cfg, decoder, _ = _real_decoder()
        from arm_planner.data_free.latent import sample_sphere
        codes = sample_sphere(32, cfg.latent_dim, generator=torch.Generator().manual_seed(3)).to(
            next(decoder.parameters()).device)
        t0 = time.time()
        o = graph_vs_grid(decoder, codes, cfg, generator=torch.Generator().manual_seed(0))
        print(f"  real decoder at init, 32 codes ({time.time() - t0:.2f}s):",
              {k: v for k, v in o.items() if k not in ("graph_counts", "grid_counts")})
    a, b = _run("smooth", _sphere_code(0)), _run("smooth", _sphere_code(1))
    print("  subspace_stability smooth/smooth:", subspace_stability(a, b))
    print("  summarise:", summarise([a, b, _run("pose", _sphere_code())]))
