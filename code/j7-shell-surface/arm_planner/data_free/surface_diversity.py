"""Shared stratified SDF measurement and symmetric closest-surface matching for J7."""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
import torch
from scipy.spatial import cKDTree

from .field import field_and_grad
from .latent import chord
from .losses import _TARGET_FLOOR


def shell_probes(count, radius, generator):
    """The tested 5/50/20/25 allocation, with weights for a uniform-volume SDF mean."""
    limits = (0., .20, .55, .70, radius)
    counts = (round(.05 * count), count // 2, round(.20 * count), 0)
    counts = (*counts[:3], count - sum(counts))
    if min(counts) < 1 or radius <= limits[-2]:
        raise ValueError("shell diversity requires four populated radial strata and radius > 0.70")
    directions = torch.nn.functional.normalize(torch.randn(count, 3, generator=generator), dim=-1)
    radii, weights = [], []
    for n, low, high in zip(counts, limits, limits[1:]):
        volume = high**3 - low**3
        radii.append((low**3 + volume * torch.rand(n, generator=generator)).pow(1 / 3))
        weights.append(torch.full((n,), volume / (radius**3 * n)))
    order = torch.randperm(count, generator=generator)
    return (directions * torch.cat(radii)[:, None])[order], torch.cat(weights)[order]


def surface_distances(points, landed, norms, neighbours, cfg, step):
    """Find matches on detached CPU coordinates. Differentiate distances to both endpoints."""
    n, count = points.shape[:2]
    residual = landed.detach().abs().numpy()
    lengths = norms.detach().numpy()
    valid = (residual < .02) & (lengths > .5)
    strict = (residual < .01) & (np.abs(lengths - 1.) < .10)
    start = cfg.surface_start_queries
    cap = min(count, start * 2**min(ceil_log2(count, start), step // cfg.surface_growth_interval))
    targets = [np.flatnonzero(row) for row in valid]
    queries = []
    for ids, accurate in zip(targets, strict):
        base = ids[:start]
        more = ids[start:][accurate[ids[start:]]]
        queries.append(np.concatenate((base, more[:max(0, cap - len(base))])))
    coordinates = points.detach().numpy()
    links = np.zeros((n, n), dtype=bool)
    links[np.arange(n)[:, None], neighbours.numpy()] = True
    links |= links.T.copy()
    links &= np.outer(valid.any(1), valid.any(1))
    flat = coordinates.reshape(-1, 3)
    query_ids = [src * count + ids for src, ids in enumerate(queries)]
    query_counts = np.array([len(ids) for ids in queries])

    def match_target(dst):
        sources = np.flatnonzero(links[:, dst])
        if not len(sources):
            return None
        tree = cKDTree(coordinates[dst, targets[dst]])
        ids = np.concatenate([query_ids[src] for src in sources])
        _, nearest = tree.query(flat[ids], k=1, workers=1)
        return ids, dst * count + targets[dst][nearest], sources * n + dst, query_counts[sources]

    # Each target has an independent index. map preserves the original target order;
    # one thread per independent query stream avoids spawning threads for every tree.
    workers = min(3, torch.get_num_threads(), n)
    if workers == 1:
        matches = list(map(match_target, range(n)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            matches = list(executor.map(match_target, range(n)))
    matches = [entry for entry in matches if entry is not None]
    matrix = points.new_zeros(n*n)
    if matches:
        source, target, edges, counts = (np.concatenate(arrays) for arrays in zip(*matches))
        p = points.reshape(-1, 3)
        delta = p[source] - p[target]
        means = torch.segment_reduce(delta.norm(dim=-1), "mean", lengths=torch.from_numpy(counts))
        matrix = matrix.scatter(0, torch.from_numpy(edges), means)
    matrix = matrix.reshape(n, n)
    return (.5 * (matrix + matrix.T)).gather(1, neighbours), {
        "surface/query_mean": float(query_counts.mean()), "surface/query_min": int(query_counts.min()),
        "surface/query_max": int(query_counts.max()), "surface/query_cap": cap,
        "surface/valid_mean": float(valid.sum(1).mean()), "surface/strict_mean": float(strict.sum(1).mean()),
        "surface/pair_coverage": float(links[np.arange(n)[:, None], neighbours.numpy()].mean())}


def ceil_log2(count, start):
    return max(0, ((count + start - 1) // start - 1).bit_length())


def field_distances(fields, gradients, x, weight, near, pose_invariant):
    """Mean absolute SDF difference after the weighted J7 pose correction for each ordered pair."""
    others = fields[near].transpose(1, 2)  # [shape, probe, neighbour]
    delta = fields.unsqueeze(-1) - others
    if pose_invariant:
        # Translation and rotation depend only on the owner shape. Project all its
        # neighbours together; only the seventh (scale) basis column varies by pair.
        rigid = torch.cat((gradients, torch.linalg.cross(x.expand_as(gradients), gradients)), -1)
        weighted_rigid_t = (weight[None, :, None] * rigid).transpose(1, 2)
        rigid_gram = weighted_rigid_t @ rigid
        scale = .5 * (fields.unsqueeze(-1) + others).detach() - (gradients * x).sum(-1, keepdim=True)
        cross = (weighted_rigid_t @ scale).transpose(1, 2)
        n, k = near.shape
        gram = fields.new_empty(n, k, 7, 7)
        gram[:, :, :6, :6] = rigid_gram[:, None]
        gram[:, :, :6, 6] = cross
        gram[:, :, 6, :6] = cross
        gram[:, :, 6, 6] = (scale.square() * weight[None, :, None]).sum(1)
        ridge = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12) * 1e-4
        gram = gram + torch.eye(7, dtype=fields.dtype, device=fields.device) * ridge[:, :, None, None]
        rhs_rigid = (weighted_rigid_t @ delta).transpose(1, 2)
        rhs_scale = (scale * delta * weight[None, :, None]).sum(1)
        rhs = torch.cat((rhs_rigid, rhs_scale.unsqueeze(-1)), -1)
        coeff = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
        delta = delta - rigid @ coeff[:, :, :6].transpose(1, 2) - scale * coeff[:, :, 6].unsqueeze(1)
    return (delta.abs() * weight[None, :, None]).sum(1)


_CPU_SCORER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="surface-score")


def _score_on_cpu(fields, x, weight, gradients, landed, codes, cfg, step):
    """All tensor inputs stay on CPU. CUDA work continues under the trainer's GPU guard."""
    begin = perf_counter()
    with torch.enable_grad():
        fields.requires_grad_()
        points = (x - fields.detach()[..., None] * gradients).requires_grad_()
        raw = torch.cdist(fields.detach() * weight, fields.detach() * weight, p=1)
        raw.fill_diagonal_(torch.inf)
        near = raw.topk(min(cfg.nearest_k, len(codes)-1), largest=False).indices
        match_start = perf_counter()
        surface, stats = surface_distances(points, landed, gradients.norm(dim=-1), near, cfg, step)
        stats["time_ms/surface_matching"] = (perf_counter() - match_start) * 1000
        field = field_distances(fields, gradients, x, weight, near, cfg.diversity_pose_invariant)
        distance = field + cfg.surface_match_weight * surface
        soft = (distance + _TARGET_FLOOR).pow(-cfg.nearest_power).sum(1).pow(-1/cfg.nearest_power)
        term = -(soft / cfg.diversity_slope + getattr(cfg, "repel_eps", .001)).log().mean()
        df, dp = torch.autograd.grad(cfg.variety_weight * term, (fields, points), allow_unused=True)
        dp = torch.zeros_like(points) if dp is None else dp
    latent = chord(codes[:, None], codes[near])
    ratio = distance.detach() / (cfg.diversity_slope * latent).clamp_min(1e-6)
    info = dict(mean_ratio=ratio.mean(), dead_zone_fraction=(ratio >= 1).float().mean(),
                mean_shape_distance=distance.detach().mean(), mean_latent_distance=latent.mean(),
                phase2=0., order_violations=0., nearest_distance=distance.detach().min(1).values.mean())
    radii = x.norm(dim=-1)
    stats.update({
        "surface/field_distance": float(field.detach().mean()),
        "surface/match_distance": float(surface.detach().mean()),
        "surface/point_gradient_norm": float(dp.norm()),
        "surface/fixed_direction": int(cfg.surface_fixed_direction),
        "surface/sdf_probes": len(x),
        "surface/shell_probes": int(((radii >= .20) & (radii < .55)).sum()),
    })
    landings = None
    if cfg.surface_fixed_direction:
        # p = x - f(x) * stop_gradient(g). Project the point force onto its source SDF.
        match_df = -(dp * gradients).sum(-1)
        stats["surface/match_source_gradient_norm"] = float(match_df.norm())
        replay_gradient = df + match_df
        if cfg.field_moves_surface:
            landings = points.detach()
    else:
        replay_gradient = dp
    stats["time_ms/diversity_scores"] = (perf_counter() - begin) * 1000
    return df, replay_gradient, landings, term.detach(), info, stats


class SurfaceDiversity:
    """Cache the small CPU loss graph, then replay its exact model derivatives in GPU chunks."""
    # Measured on the 16 GB training GPU. 35/35/34 retains the minimum
    # three gradient passes with more headroom than a 48/48/8 split.
    lookup_codes_per_batch = 104
    gradient_codes_per_batch = 35

    def __init__(self, model, codes, cfg, generator, step, *, defer_score=False):
        self.model, self.codes, self.cfg = model, codes.detach(), cfg
        x, weight = shell_probes(cfg.surface_sdf_probes, cfg.bound_radius, generator)
        self.x = x.to(codes.device)
        shape = (len(codes), len(x))
        fields, landed = torch.empty(shape), torch.empty(shape)
        gradients = torch.empty((*shape, 3))
        begin = perf_counter()
        with torch.no_grad():
            for rows, cols, mod, coords in self.blocks(self.lookup_codes_per_batch):
                f, g = field_and_grad(model, mod, coords, create_graph=False)
                p = coords - f[..., None] * g
                fields[rows, cols] = f.cpu()
                gradients[rows, cols] = g.cpu()
                landed[rows, cols] = model.query(mod, p).cpu()
        evaluated = perf_counter()
        self.stats = {"time_ms/diversity_field": (evaluated - begin) * 1000}
        self.pending_score = None
        args = (fields, x, weight, gradients, landed, codes.detach().cpu(), cfg, step)
        if defer_score:
            self.pending_score = _CPU_SCORER.submit(_score_on_cpu, *args)
        else:
            self._apply_score(_score_on_cpu(*args))

    def _apply_score(self, result):
        df, replay_gradient, landings, value, self.info, stats = result
        self.stats.update(stats)
        device = self.codes.device
        self.df = df.to(device)
        if self.cfg.surface_fixed_direction:
            self.source_df = replay_gradient.to(device)
            if landings is not None:
                self.landings = landings.to(device)
        else:
            self.dp = replay_gradient.to(device)
        self.value = value.to(device)

    def finish_score(self):
        if self.pending_score is not None:
            begin = perf_counter()
            self._apply_score(self.pending_score.result())
            self.pending_score = None
            self.stats["time_ms/diversity_wait"] = (perf_counter() - begin) * 1000

    def blocks(self, size):
        for i in range(0, len(self.codes), size):
            rows = slice(i, i+size)
            z = self.codes[rows]
            for j in range(0, len(self.x), 2048):
                cols = slice(j, j+2048)
                yield rows, cols, self.model.prepare(z), self.x[cols].expand(len(z), -1, -1)

    def backward(self):
        if self.cfg.surface_fixed_direction:
            for rows, cols, mod, coords in self.blocks(self.gradient_codes_per_batch):
                samples = coords
                upstream = self.source_df[rows, cols]
                if self.cfg.field_moves_surface:
                    # J7's independent scalar-field term also reaches the value at
                    # the fixed landing. It receives df, without the match force.
                    samples = torch.cat((samples, self.landings[rows, cols]), dim=1)
                    upstream = torch.cat((upstream, self.df[rows, cols]), dim=1)
                self.model.query(mod, samples).backward(upstream)
            return
        for rows, cols, mod, coords in self.blocks(self.gradient_codes_per_batch):
            f, g = field_and_grad(self.model, mod, coords, create_graph=True)
            p = coords - f[..., None] * g
            df = self.df[rows, cols]
            value = f
            if self.cfg.field_moves_surface:
                # Same derivative as J7's f_move: the landing coordinate is fixed in this path.
                value = f + self.model.query(mod, p.detach())
            torch.autograd.backward((value, p), (df, self.dp[rows, cols]))
