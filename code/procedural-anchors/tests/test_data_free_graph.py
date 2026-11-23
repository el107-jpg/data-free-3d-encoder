"""Tests of the data-free point graph: spatial primitives, graph findings, action losses.

Inputs are built from analytic signed distance fields (positive inside) with
exact gradients, exactly as a training step would hand them over: uniform box
samples with ``x[:, 0]`` the origin and one-step guesses ``xp = x − f·g``.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from arm_planner.data_free import analytic_shapes as A
from arm_planner.data_free import graph as graph_module
from arm_planner.data_free import spatial
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.graph import build_graph
from arm_planner.data_free.graph_losses import (
    ACTION_KEYS,
    DecoderField,
    action_losses,
    confirm,
    fragment_actions,
    hard_negatives,
    segment_points,
)
from arm_planner.data_free.spatial import (
    connected_components,
    pairs_within,
    point_segment_distance,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
E = 0.005          # consistency error handed to the graph
N = 2048
SEEDS = (0, 1, 2)


@pytest.fixture(scope="module")
def cfg():
    return DataFreeConfig()


@pytest.fixture(scope="module")
def shapes():
    return A.shapes()


def build(shapes, cfg, name, seed=0, device="cpu", mode="full", e=E):
    inputs = A.sample_inputs([shapes[name]], N, seed, device)
    return build_graph(inputs, e, cfg, mode), inputs


def counts(graph, i=0):
    in_shape = graph.frag_shape == i
    return dict(
        pieces=int(graph.num_pieces[i]),
        kind0=int((in_shape & (graph.frag_kind == 0)).sum()),
        kind1=int((in_shape & (graph.frag_kind == 1)).sum()),
        cavities=int((graph.cav_shape == i).sum()),
        gaps=int((graph.gap_shape == i).sum()),
        walls=int((graph.wall_shape == i).sum()),
    )


# =========================================================================== spatial


def brute_pairs(a, sa, b, sb, radius, unordered=False, ar=None, br=None):
    dist = torch.cdist(a.double(), b.double())
    ok = (sa[:, None] == sb[None, :]) & (dist <= radius)
    if ar is not None or br is not None:
        thr = (ar[:, None] if ar is not None else 0) + (br[None, :] if br is not None else 0)
        ok &= dist <= thr
    if unordered:
        ok &= torch.arange(a.shape[0])[:, None] < torch.arange(b.shape[0])[None, :]
    return {tuple(p) for p in ok.nonzero().tolist()}


def as_set(ia, ib, unordered=False):
    pairs = torch.stack([ia, ib], dim=1).cpu()
    if unordered:
        pairs = torch.stack([pairs.amin(dim=1), pairs.amax(dim=1)], dim=1)
    out = {tuple(p) for p in pairs.tolist()}
    assert len(out) == pairs.shape[0], "a pair was returned twice"
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cells", [1, 2, 3])
def test_pairs_within_matches_brute_force(device, cells):
    gen = torch.Generator().manual_seed(1)
    a = torch.rand(400, 3, generator=gen) * 2 - 1
    b = torch.rand(700, 3, generator=gen) * 2 - 1
    sa = torch.randint(0, 4, (400,), generator=gen)
    sb = torch.randint(0, 4, (700,), generator=gen)
    for radius in (0.07, 0.3, 0.9):
        ia, ib = pairs_within(a.to(device), sa.to(device), b.to(device), sb.to(device),
                              radius, cells=cells)
        assert as_set(ia, ib) == brute_pairs(a, sa, b, sb, radius)


@pytest.mark.parametrize("device", DEVICES)
def test_pairs_within_unordered_and_distances(device):
    gen = torch.Generator().manual_seed(2)
    a = torch.rand(900, 3, generator=gen) * 2 - 1
    a[100:120] = a[0:20]  # exact duplicates: zero distance, same cell
    sa = torch.randint(0, 3, (900,), generator=gen)
    sa[100:120] = sa[0:20]
    ad, sd = a.to(device), sa.to(device)
    for radius in (0.05, 0.25):
        ia, ib, dist = pairs_within(ad, sd, ad, sd, radius, unordered=True, return_distance=True)
        assert as_set(ia, ib, unordered=True) == brute_pairs(a, sa, a, sa, radius, unordered=True)
        assert torch.allclose(dist, (ad[ia] - ad[ib]).norm(dim=-1), atol=1e-6)
        assert bool((ia != ib).all())


@pytest.mark.parametrize("device", DEVICES)
def test_pairs_within_per_point_radii_max_pairs_and_chunks(device, monkeypatch):
    gen = torch.Generator().manual_seed(3)
    a = torch.rand(300, 3, generator=gen) * 2 - 1
    b = torch.rand(500, 3, generator=gen) * 2 - 1
    sa = torch.randint(0, 2, (300,), generator=gen)
    sb = torch.randint(0, 2, (500,), generator=gen)
    ar = torch.rand(300, generator=gen) * 0.2
    br = torch.rand(500, generator=gen) * 0.2
    args = (a.to(device), sa.to(device), b.to(device), sb.to(device), 0.4)
    ia, ib = pairs_within(*args, a_radius=ar.to(device), b_radius=br.to(device))
    expected = brute_pairs(a, sa, b, sb, 0.4, ar=ar, br=br)
    assert as_set(ia, ib) == expected
    # the chunked expansion gives the same pairs
    monkeypatch.setattr(spatial, "_CANDIDATE_CHUNK", 1000)
    ia, ib = pairs_within(*args, a_radius=ar.to(device), b_radius=br.to(device))
    assert as_set(ia, ib) == expected
    # max_pairs keeps the closest ones
    ia, ib, dist = pairs_within(*args, max_pairs=50, return_distance=True)
    full = torch.cdist(a.double(), b.double())
    full[sa[:, None] != sb[None, :]] = float("inf")
    assert ia.numel() == 50
    assert torch.allclose(dist.cpu().double().sort().values, full.flatten().sort().values[:50], atol=1e-6)
    # empty inputs
    ia, ib = pairs_within(args[0][:0], args[1][:0], args[2], args[3], 0.4)
    assert ia.numel() == 0 and ib.numel() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_point_segment_distance_matches_brute_force(device):
    gen = torch.Generator().manual_seed(4)
    p, a, b = (torch.rand(200, 3, generator=gen, dtype=torch.float64).to(device) for _ in range(3))
    b[:5] = a[:5]  # zero-length segments
    dist, t = point_segment_distance(p, a, b)
    s = torch.linspace(0, 1, 4001, dtype=torch.float64, device=device)
    on_segment = a[:, None, :] + (b - a)[:, None, :] * s[None, :, None]
    brute = (p[:, None, :] - on_segment).norm(dim=-1).amin(dim=1)
    assert torch.allclose(dist, brute, atol=1e-4)
    assert bool(((t >= 0) & (t <= 1)).all())
    closest = a + t[:, None] * (b - a)
    assert torch.allclose((p - closest).norm(dim=-1), dist, atol=1e-12)
    assert torch.allclose(dist[:5], (p[:5] - a[:5]).norm(dim=-1))


def union_find_labels(n, ea, eb):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for u, v in zip(ea, eb):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)
    return [find(i) for i in range(n)]


@pytest.mark.parametrize("device", DEVICES)
def test_connected_components_matches_union_find(device):
    gen = torch.Generator().manual_seed(5)
    cases = []
    for n, m in ((1, 0), (50, 0), (200, 80), (2000, 1500), (2000, 6000)):
        cases.append((n, torch.randint(0, n, (m,), generator=gen), torch.randint(0, n, (m,), generator=gen)))
    # a long path in scrambled order (worst case for plain label propagation)
    perm = torch.randperm(3000, generator=gen)
    cases.append((3000, perm[:-1], perm[1:]))
    # a 2-D grid with random node numbering
    side = 40
    idx = torch.randperm(side * side, generator=gen).view(side, side)
    cases.append((side * side, torch.cat([idx[:, :-1].flatten(), idx[:-1].flatten()]),
                  torch.cat([idx[:, 1:].flatten(), idx[1:].flatten()])))
    for n, ea, eb in cases:
        labels, rounds = connected_components(n, ea.to(device), eb.to(device), return_iterations=True)
        assert labels.tolist() == union_find_labels(n, ea.tolist(), eb.tolist())
        assert rounds <= 40  # pointer jumping: far fewer rounds than the diameter


# =========================================================================== cut tests


@pytest.mark.parametrize("device", DEVICES)
def test_cut_tests_match_brute_force(device):
    """The hash-grid, node-based and dense cut tests agree with direct evaluation."""
    gen = torch.Generator().manual_seed(6)
    n_shapes, per, n_edges = 3, 600, 4000
    pts2 = (torch.rand(n_shapes, per, 3, generator=gen) * 2 - 1).to(device)
    pts = pts2.reshape(-1, 3)
    shape = torch.arange(n_shapes, device=device).repeat_interleave(per)
    ea = torch.randint(0, n_shapes * per, (n_edges,), generator=gen).to(device)
    off = torch.randint(1, per, (n_edges,), generator=gen).to(device)
    eb = shape[ea] * per + (ea - shape[ea] * per + off) % per       # same shape, different node
    length = (pts[ea] - pts[eb]).norm(dim=-1)
    keep = length < 0.5
    ea, eb, length = ea[keep], eb[keep], length[keep]
    radius = (torch.rand(n_shapes * per, generator=gen) * 0.2).to(device)
    radius[torch.rand(n_shapes * per, generator=gen).to(device) < 0.85] = 0.0
    tiers = [0.025, 0.05, 0.1, 0.2]
    normal2 = torch.nn.functional.normalize(torch.randn(n_shapes, per, 3, generator=gen), dim=-1).to(device)
    surfel2 = (torch.rand(n_shapes, per, generator=gen) < 0.5).to(device)
    tol, cos_cut = 0.05, 0.6

    # brute force
    dist, t = point_segment_distance(pts2[shape[ea]], pts[ea][:, None], pts[eb][:, None])
    ref_ball = (dist < radius.view(n_shapes, per)[shape[ea]]).any(dim=1)
    u = (pts[eb] - pts[ea]) / length[:, None]
    along = t * length[:, None]
    ref_surf = (surfel2[shape[ea]] & (dist < tol) & (along >= tol) & (along <= length[:, None] - tol)
                & ((normal2[shape[ea]] * u[:, None]).sum(-1).abs() >= cos_cut)).any(dim=1)
    assert 0.05 < ref_ball.float().mean() < 0.95 and 0.005 < ref_surf.float().mean() < 0.98

    b_idx = (radius > 0).nonzero().squeeze(1)
    ball = graph_module._ball_cut(pts[ea], pts[eb], length, shape[ea], pts[b_idx], shape[b_idx],
                                  radius[b_idx], tiers)
    assert torch.equal(ball, ref_ball)
    f_idx = surfel2.reshape(-1).nonzero().squeeze(1)
    surf = graph_module._surfel_cut(pts[ea], pts[eb], length, shape[ea], pts[f_idx], shape[f_idx],
                                    normal2.reshape(-1, 3)[f_idx], tol, cos_cut)
    assert torch.equal(surf, ref_surf)
    d_ball, d_surf = graph_module._dense_cuts(pts[ea], pts[eb], length, shape[ea], pts2,
                                              radius.view(n_shapes, per), surfel2, normal2, tol, cos_cut)
    assert torch.equal(d_ball, ref_ball) and torch.equal(d_surf, ref_surf)
    # node-based C1 on the short edges
    short = length < 0.2
    nodes = torch.arange(n_shapes * per, device=device)
    relation = graph_module._node_ball_relation(nodes, pts, shape, n_shapes * per, pts[b_idx],
                                                shape[b_idx], radius[b_idx], tiers, 0.1)
    by_nodes = graph_module._ball_cut_by_nodes(relation, ea[short], eb[short], pts, pts[b_idx], radius[b_idx])
    assert torch.equal(by_nodes, ref_ball[short])


# =========================================================================== analytic shapes


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_single_sphere_is_one_clean_piece(cfg, shapes, device, seed):
    graph, _ = build(shapes, cfg, "sphere", seed, device)
    assert counts(graph) == dict(pieces=1, kind0=0, kind1=0, cavities=0, gaps=0, walls=0)
    # every solid node carries the main label, every other point -1
    solid = graph.inside | graph.band
    assert bool((graph.solid_label[solid] == graph.main_label[0]).all())
    assert bool((graph.solid_label[~solid] == -1).all())
    assert int(graph.main_label[0]) == int(graph.solid_label[0, 0])  # the origin's component


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_two_separated_spheres_give_a_fragment_with_its_joining_segment(cfg, shapes, device, seed):
    graph, (x, f, g, xp, fp) = build(shapes, cfg, "two_spheres", seed, device)
    c = counts(graph)
    assert (c["pieces"], c["kind0"], c["cavities"], c["gaps"], c["walls"]) == (2, 1, 0, 0, 0)
    row = int((graph.frag_kind == 0).nonzero()[0])
    main, other = A.sphere([-0.35, 0, 0], 0.4), A.sphere([0.55, 0, 0], 0.3)
    # true gap between the spheres is 0.2; the closest sampled pair is a bit longer
    assert 0.2 - 1e-4 <= float(graph.frag_gap[row]) <= 0.25
    assert torch.allclose((graph.frag_a[row] - graph.frag_b[row]).norm(), graph.frag_gap[row], atol=1e-6)
    delta = float(graph.delta[0])
    assert abs(float(other(graph.frag_a[row]))) <= delta + 1e-6   # end on the fragment's surface
    assert abs(float(main(graph.frag_b[row]))) <= delta + 1e-6    # end on the main body's surface
    assert abs(float(graph.frag_size[row]) - float(other(x[0]).max())) < 1e-6
    # the mask is exactly the solid nodes of the second sphere
    pts = torch.cat([x[0], xp[0]])
    solid = (graph.inside | graph.band)[0]
    in_other = solid & (other(pts) > main(pts))
    assert torch.equal(graph.frag_mask[row], in_other)
    assert fragment_actions(graph, cfg).tolist() == [True]  # gap 0.2 > delete_gap: delete


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_shell_has_one_cavity_and_is_one_piece(cfg, shapes, device, seed):
    graph, (x, f, g, xp, fp) = build(shapes, cfg, "shell", seed, device)
    c = counts(graph)
    assert (c["pieces"], c["kind0"], c["cavities"], c["gaps"], c["walls"]) == (1, 0, 1, 0, 0)
    assert int(graph.num_cavities[0]) == 1
    # the origin is in free space here: the main body is the largest component
    assert not bool((graph.inside | graph.band)[0, 0])
    # outer and inner surface are the same component
    pts = torch.cat([x[0], xp[0]])
    radius = pts.norm(dim=-1)
    guess = torch.arange(2 * N, device=pts.device) >= N
    inner = guess & graph.band[0] & ((radius - 0.4).abs() < 1e-3)
    outer = guess & graph.band[0] & ((radius - 0.8).abs() < 1e-3)
    assert int(inner.sum()) > 50 and int(outer.sum()) > 500
    # (a few surface nodes with no neighbour in reach stay single band nodes;
    # they carry no solid and are not reported as fragments)
    assert float((graph.solid_label[0, inner] == graph.main_label[0]).float().mean()) > 0.95
    assert float((graph.solid_label[0, outer] == graph.main_label[0]).float().mean()) > 0.95
    assert bool((graph.inside[0] & ~graph.band[0] & (graph.solid_label[0] != graph.main_label[0])).sum() == 0)
    # the cavity holds exactly the confident outside points inside the inner sphere
    expected = graph.outside[0] & (radius < 0.4)
    assert torch.equal(graph.cav_mask[0], expected)
    assert float(graph.cav_a[0].norm()) < 0.4 and float(graph.cav_b[0].norm()) > 0.8


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_thick_torus_is_one_piece_without_cavity(cfg, shapes, device, seed):
    graph, _ = build(shapes, cfg, "torus", seed, device)
    c = counts(graph)
    # the handle is invisible to the graph by design
    assert (c["pieces"], c["kind0"], c["cavities"], c["gaps"], c["walls"]) == (1, 0, 0, 0, 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_two_spheres_with_narrow_gap_are_a_fragment_not_a_gap_site(cfg, shapes, device, seed):
    assert 0.05 < cfg.min_feature  # the gap of this shape is 0.05
    graph, _ = build(shapes, cfg, "two_spheres_near", seed, device)
    c = counts(graph)
    assert (c["pieces"], c["kind0"], c["kind1"], c["gaps"]) == (2, 1, 0, 0)
    assert 0.05 - 1e-4 <= float(graph.frag_gap[0]) <= 0.16
    assert fragment_actions(graph, cfg).tolist() == [False]  # large and close: bridge


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_slot_in_one_piece_gives_gap_sites(cfg, shapes, device, seed):
    graph, _ = build(shapes, cfg, "slotted_box", seed, device)
    c = counts(graph)
    assert (c["pieces"], c["kind0"], c["walls"]) == (1, 0, 0)
    assert c["gaps"] >= 3
    mid, length = graph.gap_mid, graph.gap_len
    # midpoints lie in the slot (0.125 < y < 0.175, x >= -0.25): free space,
    # or exactly on the end wall of the slot for pairs along its two corners
    assert bool(((mid[:, 1] - 0.15).abs() < 0.025).all()) and bool((mid[:, 0] > -0.25 - 1e-5).all())
    value = shapes["slotted_box"](mid)
    assert bool((value < 1e-6).all()) and float((value < -1e-3).float().mean()) > 0.7
    # lengths: the slot is 0.05 wide; band nodes sit within δ of each wall
    delta = float(graph.delta[0])
    assert bool((length >= 0.05 - 2 * delta).all()) and bool((length < cfg.min_feature).all())
    assert abs(float(length.min()) - 0.05) < 0.012


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_thin_plate_gives_wall_sites(cfg, shapes, device, seed):
    graph, _ = build(shapes, cfg, "plate", seed, device)
    c = counts(graph)
    assert (c["pieces"], c["kind0"], c["cavities"], c["gaps"]) == (1, 0, 0, 0)
    assert c["walls"] == cfg.max_sites
    mid, length = graph.wall_mid, graph.wall_len
    # inside the plate; a few sites join the two rims of the edge faces and
    # have their midpoint exactly on that face
    value = shapes["plate"](mid)
    assert bool((value > -1e-6).all()) and float((value > 1e-3).float().mean()) > 0.5
    assert bool((mid[:, 2].abs() < 0.02).all())
    # sites are spread out: no two in the same place
    assert float(torch.pdist(mid).min()) > 1e-3
    delta = float(graph.delta[0])
    # thickness 0.04; ends are band nodes, each within δ of its face
    assert bool((length >= 0.04 - 2 * delta).all()) and bool((length < cfg.min_feature).all())
    assert abs(float(length.median()) - 0.04) < 0.012


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_thick_box_has_no_thin_sites_even_at_its_sharp_edges(cfg, shapes, device, seed):
    graph, (x, f, g, xp, fp) = build(shapes, cfg, "box", seed, device)
    assert counts(graph) == dict(pieces=1, kind0=0, kind1=0, cavities=0, gaps=0, walls=0)
    # the test is meaningful: many surface nodes sit on the 90° edges
    half = xp.new_tensor([0.5, 0.4, 0.3])
    on_face = ((xp[0].abs() - half).abs() < 1e-4).sum(dim=-1)
    assert int((on_face >= 2).sum()) > 200
    # and pairs across an edge do show up once the facing-normal condition and
    # the alignment threshold are relaxed, so those two conditions keep them out
    relaxed = dataclasses.replace(cfg, facing_cos=0.5, cos_cut=0.5)
    assert int(build_graph((x, f, g, xp, fp), E, relaxed).wall_shape.numel()) > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_dumbbell_is_one_piece(cfg, shapes, device, seed):
    assert 2 * 0.15 > cfg.min_feature  # neck diameter
    graph, _ = build(shapes, cfg, "dumbbell", seed, device)
    c = counts(graph)
    # A kind-1 fragment (lobe not certified through the neck) is allowed: it
    # only ever asks for a bridge. A kind-0 fragment would be wrong.
    assert (c["pieces"], c["kind0"], c["cavities"], c["gaps"], c["walls"]) == (1, 0, 0, 0, 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_sign_mode(cfg, shapes, device, seed):
    e_sign = 0.03  # between graph_full_error and graph_sign_error
    graph, _ = build(shapes, cfg, "sphere", seed, device, mode="sign", e=e_sign)
    assert counts(graph) == dict(pieces=1, kind0=0, kind1=0, cavities=0, gaps=0, walls=0)
    assert graph.stats["surface_certified"] == 0 and graph.stats["parent_certified"] == 0
    graph, _ = build(shapes, cfg, "two_spheres", seed, device, mode="sign", e=e_sign)
    c = counts(graph)
    assert (c["pieces"], c["kind0"], c["kind1"], c["cavities"]) == (2, 1, 0, 0)
    assert 0.2 - 2 * float(graph.delta[0]) <= float(graph.frag_gap[0]) <= 0.25


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["full", "sign"])
def test_batch_equals_one_at_a_time(cfg, shapes, device, mode):
    names = list(shapes)
    e = E if mode == "full" else 0.03
    inputs = A.sample_inputs([shapes[k] for k in names], N, 7, device)
    batch = build_graph(inputs, e, cfg, mode)
    for i, name in enumerate(names):
        single = build_graph(tuple(v[i:i + 1] for v in inputs), e, cfg, mode)
        assert counts(batch, i) == counts(single, 0), name
        assert int(batch.num_cavities[i]) == int(single.num_cavities[0]), name
        for field in ("inside", "band", "outside"):
            assert torch.equal(getattr(batch, field)[i], getattr(single, field)[0]), name
        # same partition into components (labels differ by the index offset)
        offset = i * 2 * N
        for field in ("solid_label", "solid_label_pess", "outside_label"):
            b, s = getattr(batch, field)[i], getattr(single, field)[0]
            assert torch.equal(torch.where(b >= 0, b - offset, b), s), (name, field)
        fb, fs = batch.frag_shape == i, single.frag_shape == 0
        assert torch.equal(batch.frag_kind[fb], single.frag_kind[fs]), name
        assert torch.allclose(batch.frag_size[fb], single.frag_size[fs]), name
        assert torch.allclose(batch.frag_gap[fb], single.frag_gap[fs], atol=1e-6), name
        assert torch.equal(batch.frag_mask[fb], single.frag_mask[fs]), name
        assert torch.equal(batch.cav_mask[batch.cav_shape == i], single.cav_mask), name
        assert torch.allclose(batch.gap_len[batch.gap_shape == i], single.gap_len, atol=1e-6), name
        assert torch.allclose(batch.wall_len[batch.wall_shape == i], single.wall_len, atol=1e-6), name


def test_graph_ignores_non_finite_points_and_accepts_a_field_batch(cfg, shapes):
    x, f, g, xp, fp = A.sample_inputs([shapes["sphere"]], N, 0)
    xp = xp.clone()
    xp[0, 5] = float("nan")
    xp[0, 6] = 1e9
    fb = SimpleNamespace(x=x, f=f.requires_grad_(True), g=g, xp=xp, fp=fp)
    graph = build_graph(fb, torch.tensor(E), cfg)
    assert counts(graph) == dict(pieces=1, kind0=0, kind1=0, cavities=0, gaps=0, walls=0)
    assert not bool((graph.inside | graph.band | graph.outside)[0, N + 5:N + 7].any())
    assert not graph.solid_label.requires_grad


# =========================================================================== graph losses


def analytic_field(field):
    return lambda shape_index, points: field(points)


def add_fragment(graph, a, b, size=0.2):
    """Append a fake kind-0 fragment (one member point) with the given joining segment."""
    a, b = torch.tensor([a]), torch.tensor([b])
    mask = torch.zeros(1, graph.inside.shape[1], dtype=torch.bool)
    mask[0, 1] = True
    return dataclasses.replace(
        graph,
        frag_shape=torch.cat([graph.frag_shape, torch.zeros(1, dtype=torch.long)]),
        frag_kind=torch.cat([graph.frag_kind, torch.zeros(1, dtype=torch.long)]),
        frag_size=torch.cat([graph.frag_size, torch.tensor([size])]),
        frag_gap=torch.cat([graph.frag_gap, (a - b).norm(dim=-1)]),
        frag_a=torch.cat([graph.frag_a, a]), frag_b=torch.cat([graph.frag_b, b]),
        frag_mask=torch.cat([graph.frag_mask, mask]),
        num_pieces=graph.num_pieces + 1,
    )


def add_cavity(graph, a, b):
    a, b = torch.tensor([a]), torch.tensor([b])
    mask = torch.zeros(1, graph.inside.shape[1], dtype=torch.bool)
    mask[0, 2] = True
    return dataclasses.replace(
        graph,
        cav_shape=torch.cat([graph.cav_shape, torch.zeros(1, dtype=torch.long)]),
        cav_a=torch.cat([graph.cav_a, a]), cav_b=torch.cat([graph.cav_b, b]),
        cav_mask=torch.cat([graph.cav_mask, mask]),
        num_cavities=graph.num_cavities + 1,
    )


def test_confirm_drops_fake_fragment_and_keeps_real_one(cfg, shapes):
    graph, _ = build(shapes, cfg, "two_spheres")
    assert counts(graph)["kind0"] == 1
    # fake: both ends inside the main sphere, so the whole segment is inside
    fake = add_fragment(graph, [-0.5, 0.1, 0.0], [-0.2, -0.1, 0.1])
    out = confirm(analytic_field(shapes["two_spheres"]), None, fake, cfg)
    assert out.frag_shape.numel() == 1
    assert torch.equal(out.frag_a, graph.frag_a) and torch.equal(out.frag_mask, graph.frag_mask)
    assert int(out.num_pieces[0]) == 2
    assert out.stats["confirm_dropped_kind0"] == 1
    # a fake kind-1 fragment whose segment crosses free space is dropped too
    crossing = add_fragment(graph, [-0.3, 0.0, 0.0], [0.5, 0.0, 0.0])
    crossing = dataclasses.replace(crossing, frag_kind=torch.tensor([0, 1]))
    out = confirm(analytic_field(shapes["two_spheres"]), None, crossing, cfg)
    assert out.frag_kind.tolist() == [0] and out.stats["confirm_dropped_kind1"] == 1


def test_confirm_drops_fake_cavity_and_keeps_real_one(cfg, shapes):
    graph, _ = build(shapes, cfg, "shell")
    assert counts(graph)["cavities"] == 1
    out = confirm(analytic_field(shapes["shell"]), None, graph, cfg)
    assert out.cav_shape.numel() == 1 and int(out.num_cavities[0]) == 1
    # fake: a segment that stays in the free space outside the shell
    fake = add_cavity(graph, [0.85, 0.3, 0.0], [0.95, 0.9, 0.2])
    out = confirm(analytic_field(shapes["shell"]), None, fake, cfg)
    assert out.cav_shape.numel() == 1 and torch.equal(out.cav_a, graph.cav_a)
    assert out.stats["confirm_dropped_cavities"] == 1 and int(out.num_cavities[0]) == 1


class OffsetField:
    """Analytic field plus one learnable offset; records what it returned."""

    def __init__(self, field):
        self.field = field
        self.offset = torch.zeros((), requires_grad=True)
        self.outputs = []

    def __call__(self, shape_index, points):
        out = self.field(points) + self.offset
        out.retain_grad()
        self.outputs.append(out)
        return out

    def batch(self, inputs):
        x, f, g, xp, fp = inputs
        f, fp = f + self.offset, fp + self.offset
        f.retain_grad(), fp.retain_grad()
        return SimpleNamespace(x=x, f=f, g=g, xp=xp, fp=fp)


def run_actions(shapes, cfg, name, seed=0):
    graph, inputs = build(shapes, cfg, name, seed)
    field = OffsetField(shapes[name])
    graph = confirm(analytic_field(shapes[name]), None, graph, cfg)
    fb = field.batch(inputs)
    losses = action_losses(field, None, fb, graph, cfg)
    assert tuple(losses) == ACTION_KEYS
    assert all(v.dim() == 0 and bool(v >= 0) for v in losses.values())
    return graph, field, fb, losses


def only_positive(losses, key):
    return float(losses[key]) > 0 and all(float(v) == 0 for k, v in losses.items() if k != key)


def test_action_losses_are_zero_when_nothing_is_found(cfg, shapes):
    for name in ("sphere", "box", "torus"):
        graph, field, fb, losses = run_actions(shapes, cfg, name)
        if name != "torus":
            assert graph.frag_shape.numel() == 0
        assert all(float(v) == 0 for v in losses.values()), name


def test_bridge_loss_and_gradient(cfg, shapes):
    graph, field, fb, losses = run_actions(shapes, cfg, "two_spheres_near")
    assert only_positive(losses, "bridge")
    losses["bridge"].backward()
    assert float(field.offset.grad) < 0            # descent raises the field
    seg_values = field.outputs[-1]
    seg_grad = seg_values.grad
    assert bool((seg_grad <= 0).all())
    assert bool((seg_grad[seg_values.detach() < 0] < 0).all())   # pushes the free part up
    assert bool((seg_grad[seg_values.detach() >= 0] == 0).all())  # zero margin: no thickening
    assert fb.f.grad is None or float(fb.f.grad.abs().sum()) == 0  # existing values untouched


def test_delete_loss_and_gradient(cfg, shapes):
    graph, field, fb, losses = run_actions(shapes, cfg, "two_spheres")
    assert only_positive(losses, "delete")
    losses["delete"].backward()
    assert float(field.offset.grad) > 0            # descent lowers the field
    assert not field.outputs                       # no new field evaluation at all
    row = int((graph.frag_kind == 0).nonzero()[0])
    grad = torch.cat([fb.f.grad, fb.fp.grad], dim=1)[0]
    values = torch.cat([fb.f, fb.fp], dim=1)[0].detach()
    member = graph.frag_mask[row]
    assert bool((grad[~member] == 0).all())
    assert bool((grad[member & (values > 0)] > 0).all())


def test_fill_loss_and_gradient(cfg, shapes):
    graph, field, fb, losses = run_actions(shapes, cfg, "shell")
    assert only_positive(losses, "fill")
    losses["fill"].backward()
    assert float(field.offset.grad) < 0            # descent raises the field in the cavity
    grad = torch.cat([fb.f.grad, fb.fp.grad], dim=1)[0]
    member = graph.cav_mask[0]
    assert bool((grad[~member] == 0).all()) and bool((grad[member] < 0).all())


def test_widen_loss_and_gradient(cfg, shapes):
    graph, field, fb, losses = run_actions(shapes, cfg, "slotted_box")
    assert float(losses["widen"]) > 0 and float(losses["thicken"]) == 0
    losses["widen"].backward()
    assert float(field.offset.grad) > 0            # descent makes the gap midpoints more negative
    values, grad = field.outputs[-1].detach(), field.outputs[-1].grad
    assert float((values < 0).float().mean()) > 0.7 and bool((values > -cfg.min_feature / 2).all())
    assert bool((grad[values < 0] > 0).all()) and bool((grad[values >= 0] == 0).all())


def test_thicken_loss_and_gradient(cfg, shapes):
    graph, field, fb, losses = run_actions(shapes, cfg, "plate")
    assert only_positive(losses, "thicken")
    losses["thicken"].backward()
    assert float(field.offset.grad) < 0            # descent raises the field inside the wall
    values, grad = field.outputs[-1].detach(), field.outputs[-1].grad
    assert float((values > 0).float().mean()) > 0.5
    # only sites whose midpoint is really inside the wall (deeper than a quarter of the
    # measured thickness) are acted on; rim sites with the midpoint on a face are not
    active = values > 0.25 * graph.wall_len
    assert float(active.float().mean()) > 0.5
    assert bool((grad[active] < 0).all()) and bool((grad[~active] == 0).all())


def test_hard_negatives(cfg, shapes):
    names = ["sphere", "two_spheres", "shell", "plate"]
    inputs = A.sample_inputs([shapes[k] for k in names], N, 0)
    graph = build_graph(inputs, E, cfg)
    fb = SimpleNamespace(x=inputs[0], f=inputs[1], g=inputs[2], xp=inputs[3], fp=inputs[4])
    out = hard_negatives(graph, fb, cfg, per_shape=48, generator=torch.Generator().manual_seed(0))
    assert out.shape == (4, 48, 3)
    finite = torch.isfinite(out).all(dim=-1)
    assert int(finite[0].sum()) == 0                       # nothing found on the sphere
    assert int(finite[1].sum()) == 48 and int(finite[2].sum()) == 48
    assert int(finite[3].sum()) == cfg.max_sites           # the 32 wall sites, NaN-padded
    assert bool((finite[:, 1:] <= finite[:, :-1]).all())   # valid rows first
    # the joining-segment samples are preferred
    seg = segment_points(graph.frag_a[graph.frag_shape == 1], graph.frag_b[graph.frag_shape == 1],
                         cfg.segment_samples).reshape(-1, 3)
    assert bool((torch.cdist(seg, out[1][finite[1]]).amin(dim=1) < 1e-6).all())


def test_decoder_field_adapter_and_full_pipeline_on_the_initial_decoder():
    """`(decoder, mod)` path: the initial decoder is a sphere, so nothing is found."""
    decoder_module = pytest.importorskip("arm_planner.data_free.decoder")
    field_module = pytest.importorskip("arm_planner.data_free.field")
    torch.manual_seed(0)
    small = DataFreeConfig(width=32, layers=3, skip_layer=1, mapping_width=16, points=N)
    decoder = decoder_module.ModulatedSphereDecoder(small)
    codes = torch.randn(3, small.latent_dim)
    x = torch.rand(3, N, 3) * 2 - 1
    x[:, 0] = 0
    fb = field_module.evaluate_field(decoder, codes, torch.arange(3), x, small)
    # the adapter returns the same values as a direct query
    idx = torch.tensor([2, 0, 0, 1, 2])
    points = torch.rand(5, 3) - 0.5
    adapter = DecoderField(decoder, fb.mod, 3)
    direct = torch.stack([decoder.query(fb.mod, points[None].expand(3, -1, -1))[i, j]
                          for j, i in enumerate(idx.tolist())])
    assert torch.allclose(adapter(idx, points), direct, atol=1e-6)

    e = float(fb.fp.detach().abs().median())
    graph = build_graph(fb, e, small)
    graph = confirm(decoder, fb.mod, graph, small)
    assert graph.num_pieces.tolist() == [1, 1, 1] and graph.num_cavities.tolist() == [0, 0, 0]
    assert int((graph.frag_kind == 0).sum()) == 0
    losses = action_losses(decoder, fb.mod, fb, graph, small)
    assert tuple(losses) == ACTION_KEYS
    assert all(float(v) < 1e-8 for v in losses.values())
    sum(losses.values()).backward() if any(v.requires_grad for v in losses.values()) else None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_thin_sites_are_only_reported_on_the_main_body(cfg, device, seed):
    """A detached thin piece is a fragment to delete or bridge, never a wall to thicken."""
    body = A.sphere([0.0, 0.0, 0.0], 0.4)
    thin = A.box([0.72, 0.0, 0.0], [0.12, 0.12, 0.02])          # 0.04 thick, below min_feature
    field = A.union(body, thin)
    inputs = A.sample_inputs([field], N, seed, device)
    graph = build_graph(inputs, E, cfg, mode="full")
    c = counts(graph)
    assert c["kind0"] >= 1                 # the detached plate is found as a fragment
    assert c["walls"] == 0 and c["gaps"] == 0
    # (Sites on a thin part of the main body itself are covered by test_thin_plate_gives_wall_sites.)
