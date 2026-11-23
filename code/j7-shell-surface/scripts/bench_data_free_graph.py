"""Benchmark of the data-free point graph (``arm_planner.data_free.graph.build_graph``).

B shapes (a mix of the analytic test shapes), N volume samples each. Reports
the median wall time of ``build_graph`` over repeated calls after warm-up
(CUDA synchronised), a per-stage breakdown (from a separate profiled pass, in
which every stage is synchronised), edge counts per family and state, peak GPU
memory, and the CPU time for one shape.

    PYTHONPATH=. .venv/bin/python scripts/bench_data_free_graph.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from arm_planner.data_free import analytic_shapes
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.graph import build_graph
from arm_planner.data_free.graph_losses import confirm

STAGES = [
    ("node_classing", "node classing"),
    ("surface_edges", "hash-grid surface edges (pairs within r_s)"),
    ("edge_local_rules", "edge states: certify, C2a, one-sided, sites, rescue edges"),
    ("deep_edges", "deep edges (volume + containment, dense)"),
    ("cut_tests", "cut tests C1 (outside balls) and C2b (surfel crossings)"),
    ("label_propagation", "label propagation (solid graph)"),
    ("findings_components", "findings: main body, fragment and cavity selection"),
    ("findings_closest_pairs", "findings: joining segments (closest pairs)"),
    ("findings_kind1_guard", "findings: kind-1 segment check"),
    ("findings_sites", "findings: thin sites"),
    ("outside_pairs", "outside graph: hash-grid edges (pairs within r_v)"),
    ("outside_labels", "outside graph: certified label propagation"),
    ("outside_cuts", "outside graph: cut tests and merge"),
    ("stats", "statistics"),
]


def timed(fn, device, repeats):
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1e3)
    return times, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--error", type=float, default=0.005, help="consistency error e")
    parser.add_argument("--noise", type=float, default=0.0, help="relative value noise on f and fp")
    parser.add_argument("--mode", default="full", choices=["full", "sign"])
    parser.add_argument("--shape", default=None, help="use only this analytic shape")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = DataFreeConfig(points=args.points)
    library = analytic_shapes.shapes()
    names = [args.shape] if args.shape else list(library)
    chosen = [names[i % len(names)] for i in range(args.batch)]
    device = torch.device(args.device)
    inputs = analytic_shapes.sample_inputs(
        [library[k] for k in chosen], args.points, seed=0, device=device, noise=args.noise
    )
    print(f"device {device}, B={args.batch}, N={args.points}, mode={args.mode}, e={args.error}, "
          f"r_s={cfg.r_s():.4f}, r_v={cfg.r_v():.4f}")
    print("shapes:", ", ".join(f"{k}×{chosen.count(k)}" for k in names))

    def run(profile=False):
        return build_graph(inputs, args.error, cfg, mode=args.mode, profile=profile)

    timed(run, device, args.warmup)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times, graph = timed(run, device, args.repeats)
    print(f"\nbuild_graph: median {statistics.median(times):.1f} ms, min {min(times):.1f}, "
          f"max {max(times):.1f}  ({args.repeats} calls after {args.warmup} warm-up)")
    if device.type == "cuda":
        print(f"peak GPU memory during build_graph: "
              f"{torch.cuda.max_memory_allocated(device) / 2**20:.0f} MiB allocated")

    profiles = [run(profile=True).stats["timings_ms"] for _ in range(args.repeats)]
    print("\nper-stage median (profiled pass: synchronised after every stage)")
    total = 0.0
    for key, label in STAGES:
        value = statistics.median(p.get(key, 0.0) for p in profiles)
        total += value
        print(f"  {label:58s} {value:7.2f} ms")
    print(f"  {'sum':58s} {total:7.2f} ms")

    field = lambda index, points: torch.stack(
        [library[k](points) for k in names]).gather(
            0, torch.tensor([names.index(k) for k in chosen], device=device)[index][None])[0]
    times_c, confirmed = timed(lambda: confirm(field, None, graph, cfg), device, 5)
    print(f"\nconfirm (analytic field): median {statistics.median(times_c):.1f} ms")

    stats = graph.stats
    print("\nnodes per shape (mean): " + ", ".join(
        f"{k[6:]} {stats[k] / args.batch:.0f}" for k in
        ("nodes_solid", "nodes_inside", "nodes_band", "nodes_outside", "nodes_deep", "nodes_surfel")))
    print("edges (whole batch):")
    for key in ("surface_edges", "surface_rescue_edges", "surface_certified", "surface_cut_c2a",
                "surface_cut_local", "surface_candidates", "volume_certified",
                "volume_dropped_hidden_gap", "volume_candidates", "deep_certified_edges",
                "parent_certified", "parent_unknown", "tested", "tested_cut_c1",
                "tested_cut_c2b", "tested_unknown", "untested_redundant", "outside_edges", "outside_certified", "outside_tested",
                "outside_cut_ball", "outside_cut_c2b"):
        print(f"  {key:28s} {stats[key]:>9d}")
    print("label propagation rounds: pessimistic {cc_rounds_pess}, cut-test stages "
          "{cc_rounds_stages} (in {cut_stages_run} stages)".format(**stats))
    print("findings (whole batch): kind-0 fragments {fragments_kind0}, kind-1 {fragments_kind1}, "
          "cavities {cavities}, gap sites {gap_sites}, wall sites {wall_sites}".format(**stats))
    print("after confirm: kind-0 {}, kind-1 {}, cavities {}".format(
        int((confirmed.frag_kind == 0).sum()), int((confirmed.frag_kind == 1).sum()),
        confirmed.cav_shape.numel()))

    # CPU sanity number: one shape
    cpu_inputs = analytic_shapes.sample_inputs([library[names[0]]], args.points, seed=0, device="cpu")
    cpu = torch.device("cpu")
    timed(lambda: build_graph(cpu_inputs, args.error, cfg, mode=args.mode), cpu, 2)
    times_cpu, _ = timed(lambda: build_graph(cpu_inputs, args.error, cfg, mode=args.mode), cpu, 5)
    print(f"\nCPU, one shape ({names[0]}): median {statistics.median(times_cpu):.1f} ms")


if __name__ == "__main__":
    main()
