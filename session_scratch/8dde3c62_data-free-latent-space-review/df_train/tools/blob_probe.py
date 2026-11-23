"""What keeps detached blobs alive at a checkpoint: detection, gradient balance, who pays."""
import dataclasses
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.field import evaluate_field, sample_points
from arm_planner.data_free.graph import build_graph
from arm_planner.data_free.graph_losses import action_losses, confirm, fragment_actions
from arm_planner.data_free.train import Trainer

RADIUS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.65
tr = Trainer.from_checkpoint(sys.argv[1], device="cuda", tensorboard=False)
cfg, model = tr.cfg, tr.model
params = list(model.parameters())
mult = tr.multipliers
print("step", tr.step_count, "multipliers", {k: round(v, 2) for k, v in mult.multipliers().items()})


def flat_grad(value):
    g = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
    return torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1) for t, p in zip(g, params)])


tot = {}
def add(k, v):
    tot.setdefault(k, []).append(float(v))

builds = 6
for i in range(builds):
    batch = tr.sample_batch()
    fb = evaluate_field(model, batch.codes, batch.cluster, batch.x, cfg)
    e = float(L.consistency_error(fb))
    graph = confirm(model, fb.mod, build_graph(fb, e, cfg, mode="full"), cfg)
    n = fb.x.shape[1]
    pts = torch.cat([fb.x, fb.xp], dim=1).detach()
    val = torch.cat([fb.f, fb.fp], dim=1).detach()
    blob = (val > 0) & (pts.norm(dim=-1) > RADIUS)                 # B x 2N: positive points away from the body
    covered = torch.zeros_like(blob)
    kind0 = graph.frag_kind == 0
    for row in kind0.nonzero().reshape(-1).tolist():
        covered[int(graph.frag_shape[row])] |= graph.frag_mask[row]
    has_blob = blob.any(dim=1)
    found = (blob & covered).any(dim=1)
    add("shapes with a blob point (of %d)" % blob.shape[0], has_blob.sum())
    add("  of those, shapes where a fragment mask covers a blob point", (found & has_blob).sum())
    add("blob points per shape (x and xp, of %d)" % (2 * n), blob.sum() / blob.shape[0])
    add("  of those volume samples x only", blob[:, :n].sum() / blob.shape[0])
    add("fraction of blob points inside a fragment mask", (blob & covered).sum() / blob.sum().clamp_min(1))
    add("confirmed kind-0 fragments per shape", kind0.sum() / blob.shape[0])
    delete = fragment_actions(graph, cfg)
    add("  marked delete", delete.sum() / blob.shape[0])
    add("  marked bridge", ((~delete) & kind0).sum() / blob.shape[0])
    add("mean max f in a blob (over shapes with one)", torch.where(blob, val, torch.zeros_like(val)).amax(dim=1)[has_blob].mean() if has_blob.any() else 0.0)
    add("gap sites per shape", graph.gap_shape.numel() / blob.shape[0])
    add("wall sites per shape", graph.wall_shape.numel() / blob.shape[0])
    add("wall sites farther than %.2f from the origin (fraction)" % RADIUS, (graph.wall_mid.norm(dim=-1) > RADIUS).float().mean() if graph.wall_shape.numel() else 0.0)

    acts = action_losses(model, fb.mod, fb, graph, cfg)
    div, info = L.diversity(fb, cfg, tr.step_count)
    terms = {"diversity": div, "eikonal": L.eikonal(fb), "consistency": L.consistency(fb), "barrier": L.barrier(fb, cfg)}
    terms.update(L.pose_terms(fb, cfg))
    terms.update(acts)
    for k, v in terms.items():
        w = 1.0 if k == "diversity" else mult.multiplier(k)
        add("value   " + k, v)
        if v.requires_grad and float(v.detach().abs()) > 0:
            g = flat_grad(v)
            add("|grad| unweighted " + k, g.norm())
            add("|grad| weighted   " + k, w * g.norm())
            if k != "diversity":
                gd = flat_grad(div)
                add("cos(grad, diversity grad) " + k, torch.nn.functional.cosine_similarity(g, gd, dim=0))

    # who pays: same batch with every positive value outside RADIUS pushed to free space
    far = fb.x.norm(dim=-1) > RADIUS
    f_masked = torch.where(far & (fb.f > 0), -fb.f.detach().abs() - 0.02, fb.f.detach())
    masked = dataclasses.replace(fb, f=f_masked)
    plain = dataclasses.replace(fb, f=fb.f.detach())
    d0, i0 = L.diversity(plain, cfg, tr.step_count)
    d1, i1 = L.diversity(masked, cfg, tr.step_count)
    p0, p1 = L.pose_terms(plain, cfg), L.pose_terms(masked, cfg)
    add("as is:   diversity loss", d0); add("masked:  diversity loss", d1)
    add("as is:   mean ratio", i0["mean_ratio"]); add("masked:  mean ratio", i1["mean_ratio"])
    add("as is:   dead-zone fraction", i0["dead_zone_fraction"]); add("masked:  dead-zone fraction", i1["dead_zone_fraction"])
    for k in L.POSE_KEYS:
        add("as is:   " + k, p0[k]); add("masked:  " + k, p1[k])
    _, c0, cov0 = L.moments(plain, cfg); _, c1, cov1 = L.moments(masked, cfg)
    tr0 = torch.diagonal(cov0, dim1=-2, dim2=-1).sum(-1); tr1 = torch.diagonal(cov1, dim1=-2, dim2=-1).sum(-1)
    add("as is:   trace mean (target %.4f)" % cfg.cov_trace_target, tr0.mean()); add("masked:  trace mean", tr1.mean())
    add("as is:   soft volume fraction", L.soft_inside(plain, cfg).mean()); add("masked:  soft volume fraction", L.soft_inside(masked, cfg).mean())

print(f"means over {builds} training batches (blob point: f > 0 and |p| > {RADIUS})")
for k, v in tot.items():
    print(f"  {k:62s} {sum(v) / len(v):12.4g}")
