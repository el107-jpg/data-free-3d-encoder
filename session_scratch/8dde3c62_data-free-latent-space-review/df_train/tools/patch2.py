def sub(s, a, b, tag):
    assert a in s, tag
    return s.replace(a, b)


p = 'arm_planner/data_free/diagnostics.py'
s = open(p).read()
s = sub(s, '''    mode: str = "full",
    from_rows: bool = False,
) -> dict:''', '''    mode: str = "full",
    from_rows: bool = False,
    confirmed: bool = True,
) -> dict:''', 1)
s = sub(s, '''    the same codes and returns ``compare_graph_to_grid`` plus the two sets of
    counts (``graph_counts``, ``grid_counts``) and ``consistency_error``.
    Needs ``field.py`` and ``graph.py``.
    """
    from .field import evaluate_field, sample_points
    from .graph import build_graph
''', '''    the same codes and returns ``compare_graph_to_grid`` plus the two sets of
    counts (``graph_counts``, ``grid_counts``) and ``consistency_error``.

    ``confirm`` is part of the method, so by default (``confirmed=True``) the
    graph's findings are re-checked with the decoder first and the counts are
    those after it (``confirm`` keeps ``num_pieces`` / ``num_cavities`` up to
    date). ``confirm_dropped`` reports what it removed (kind-0 fragments,
    kind-1 fragments, cavities) and ``graph_counts_raw`` the counts before.
    Needs ``field.py``, ``graph.py`` and ``graph_losses.py``.
    """
    from .field import evaluate_field, sample_points
    from .graph import build_graph
    from .graph_losses import confirm
''', 2)
s = sub(s, '''        graph = build_graph(fb, e, cfg, mode=mode)
    graph_counts = graph_topology(graph, from_rows=from_rows)
''', '''        graph = build_graph(fb, e, cfg, mode=mode)
        raw_counts = graph_topology(graph, from_rows=from_rows)
        if confirmed:
            graph = confirm(decoder, fb.mod, graph, cfg)
    graph_counts = graph_topology(graph, from_rows=from_rows)
''', 3)
s = sub(s, '''    out["grid_counts"] = grid_counts
    out["consistency_error"] = e
''', '''    out["graph_counts_raw"] = {k: v.cpu() for k, v in raw_counts.items()}
    out["confirm_dropped"] = {
        key: int(graph.stats.get(f"confirm_dropped_{key}", 0))
        for key in ("kind0", "kind1", "cavities")
    }
    out["grid_counts"] = grid_counts
    out["consistency_error"] = e
''', 4)
open(p, 'w').write(s)

p = 'arm_planner/data_free/train.py'
s = open(p).read()
s = sub(s, '''FIELD_KEYS = ("eikonal", "consistency", "barrier", "origin", "volume", "pose")''',
        '''FIELD_KEYS = ("eikonal", "consistency", "barrier", "origin", "volume") + L.POSE_KEYS''', 5)
s = sub(s, '''            "pose": L.pose(fb, cfg),
        }
''', '''        }
        # Unbiased split-sample pose terms: noisy, and slightly negative at times.
        constraints.update(L.pose_terms(fb, cfg))
''', 6)
s = sub(s, '''        f"pose {record['term/pose']:.3g}  e''',
        '''        f"pose c {record['term/pose_centroid']:.2g} s {record['term/pose_scale']:.2g} "
        f"a {record['term/pose_axes']:.2g}  e''', 7)
s = sub(s, '''            "confirm_dropped_kind0", "confirm_dropped_kind1", "confirm_dropped_cavities",
            "build_ms")}''', '''            "extra_pieces", "cavities_uncapped",
            "confirm_dropped_kind0", "confirm_dropped_kind1", "confirm_dropped_cavities",
            "build_ms")}''', 8)
s = sub(s, '''        "graph/wall_sites": int(graph.wall_shape.numel()),
''', '''        "graph/wall_sites": int(graph.wall_shape.numel()),
        # uncapped counts kept up to date by confirm (pieces beyond the main body)
        "graph/extra_pieces": int((graph.num_pieces - 1).clamp_min(0).sum())
        if graph.num_pieces is not None else 0,
        "graph/cavities_uncapped": int(graph.num_cavities.sum())
        if graph.num_cavities is not None else 0,
''', 9)
s = sub(s, '''"""Counts of what the (confirmed) graph in use reports, over the whole batch."""''',
        '''"""Counts of what the graph in use reports after ``confirm``, over the whole batch."""''', 10)
s = sub(s, '''        for key in ("pieces_agreement", "cavities_agreement", "both_agreement"):
            out[f"graph_vs_grid/{key}"] = check[key]
''', '''        for key in ("pieces_agreement", "cavities_agreement", "both_agreement"):
            out[f"graph_vs_grid/{key}"] = check[key]
        for key, dropped in check["confirm_dropped"].items():
            out[f"graph_vs_grid/confirm_dropped_{key}"] = dropped
        out["graph_vs_grid/raw_extra_pieces"] = int(
            (check["graph_counts_raw"]["pieces"] - 1).clamp_min(0).sum())
        out["graph_vs_grid/raw_cavities"] = int(check["graph_counts_raw"]["cavities"].sum())

        # pose of the probes from a large sample (the training values are noisy)
        out.update(self._probe_pose(codes, gen))
''', 11)
s = sub(s, '''        results = [
            D.dead_directions(model, code, cfg, points=int(cfg.dead_direction_points), generator=gen)
            for code in codes[: max(int(cfg.dead_direction_probes), 0)]
        ]
        for key, value in D.summarise(results).items():
            out[f"dead/{key}"] = float(value)
''', '''        results, stability, live_pair = [], [], []
        for code in codes[: max(int(cfg.dead_direction_probes), 0)]:
            here = D.dead_directions(model, code, cfg, points=int(cfg.dead_direction_points),
                                     generator=gen)
            results.append(here)
            # live subspace of a neighbour at chord cfg.dead_direction_neighbour
            direction = torch.randn(code.shape, generator=gen).to(code)
            direction = torch.nn.functional.normalize(direction - (direction @ code) * code, dim=0)
            theta = 2.0 * math.asin(min(float(cfg.dead_direction_neighbour) / 2.0, 1.0))
            neighbour = math.cos(theta) * code + math.sin(theta) * direction
            there = D.dead_directions(model, neighbour, cfg, points=int(cfg.dead_direction_points),
                                      generator=gen)
            value = D.subspace_stability(here, there)
            if math.isfinite(value):
                stability.append(value)
                live_pair.append(min(int(here["live"]), int(there["live"])))
        for key, value in D.summarise(results).items():
            out[f"dead/{key}"] = float(value)
        # Overlap of the live subspaces (1 = same, about k / (D - 1) = chance), over
        # the probes where both codes have a live direction; 0 with count 0 otherwise.
        out["dead/subspace_stability_mean"] = sum(stability) / len(stability) if stability else 0.0
        out["dead/subspace_stability_count"] = len(stability)
        out["dead/subspace_stability_chance"] = (
            sum(live_pair) / len(live_pair) / (cfg.latent_dim - 1) if live_pair else 0.0)
''', 12)
s = sub(s, '''    # ----------------------------------------------------------- checkpoints
''', '''    @torch.no_grad()
    def _probe_pose(self, codes: Tensor, generator: torch.Generator) -> Dict[str, float]:
        """Centroid, covariance trace and off-diagonals of the probes at ``diag_pose_points``."""
        cfg = self.cfg
        count = max(int(cfg.diag_pose_points), 16)
        x = (torch.rand(count, 3, generator=generator) * 2.0 - 1.0).to(self.device)
        mod = self.model.prepare(codes)
        chunks = [self.model.query(mod, x[None, i:i + 65536].expand(codes.shape[0], -1, -1))
                  for i in range(0, count, 65536)]
        fake = SimpleNamespace(f=torch.cat(chunks, dim=1), x=x[None].expand(codes.shape[0], -1, -1))
        _, centroid, cov = L.moments(fake, cfg)
        offset = centroid.norm(dim=-1)
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)
        off = cov - torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
        target = float(cfg.cov_trace_target)
        return {
            "pose/centroid_offset_mean": float(offset.mean()),
            "pose/centroid_offset_max": float(offset.max()),
            "pose/centroid_sq_mean": float(offset.square().mean()),
            "pose/trace_mean": float(trace.mean()),
            "pose/trace_rel_error_rms": float(((trace - target) / target).square().mean().sqrt()),
            "pose/trace_rel_error_max": float(((trace - target) / target).abs().max()),
            "pose/scale_sq_mean": float((trace - target).square().mean()),
            "pose/offdiag_abs_max": float(off.abs().max()),
            "pose/axes_sq_mean": float(off.square().sum(dim=(-2, -1)).mean()),
        }

    # ----------------------------------------------------------- checkpoints
''', 13)
s = sub(s, '''from dataclasses import dataclass
''', '''from dataclasses import dataclass
from types import SimpleNamespace
''', 14)
open(p, 'w').write(s)

p = 'arm_planner/data_free/config.py'
s = open(p).read()
s = sub(s, '''    dead_direction_points: int = 4096
''', '''    dead_direction_points: int = 4096
    dead_direction_neighbour: float = 0.1  # chord to the code used for subspace stability
    diag_pose_points: int = 262144  # samples for the probes' measured pose
''', 15)
open(p, 'w').write(s)
print("patched")
