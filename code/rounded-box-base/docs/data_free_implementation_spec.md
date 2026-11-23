# Data-free latent space: implementation spec

Design source: `Data-free latent space.md` (repo root). Read it first. This file
fixes the module layout, interfaces and algorithms so the modules can be written
independently and fit together. Where this spec is more specific than the plan,
this spec wins; where it is silent, follow the plan.

## Ground rules

- New code lives in `arm_planner/data_free/` (new package), `scripts/`, and
  `tests/test_data_free_*.py`. Do **not** edit any existing file, and do not put
  anything in `arm_planner/object_geometry/` (its `*.py` files are hashed into a
  cache signature).
- Dependencies: `torch` and `numpy` only. `scipy` / `scikit-image` may be
  imported lazily inside `diagnostics.py` and in tests.
- Self-contained: do not import from `arm_planner.object_geometry`.
- Everything in the training path is batched and vectorised on the GPU. Python
  loops over points, edges or shapes are not acceptable there. Loops over a
  small capped number of rare items (fragments, cavities) are fine.
- Sign convention: **positive inside**, negative in free space. The spatial
  gradient `g = ∇_x f` therefore points *into* the solid.
- Coordinates: shapes live in the box `[-1, 1]^3`.
- Run things from the worktree root with
  `PYTHONPATH=. /home/eric-luo/Downloads/pointcloud-motion-planning/.venv/bin/python`
  (torch 2.7, CUDA, one RTX 5080 16 GB shared with other agents: keep GPU memory
  under ~4 GB and do not leave processes running).
- Tests: `... -m pytest tests/test_data_free_<name>.py -q`. Tests must run in
  seconds, on CPU by default, GPU only when `torch.cuda.is_available()`.
- Do not `git commit`, `git stash`, or touch git state. The orchestrator commits.
- Report measured numbers only. If something was not run, say so.

## Files and owners

| File | Contents | Task |
|---|---|---|
| `config.py` | `DataFreeConfig` dataclass | T1 |
| `latent.py` | sphere sampling, chord distance, slerp, clusters, tangent basis | T1 |
| `decoder.py` | `ModulatedSphereDecoder` | T1 |
| `field.py` | `FieldBatch`, `evaluate_field` | T1 |
| `losses.py` | SDF terms, validity, pose, descriptors, diversity | T2 |
| `multipliers.py` | `AdaptiveMultipliers` | T2 |
| `spatial.py` | hash-grid neighbour search, label propagation | T3 |
| `graph.py` | `ShapeGraph`, `build_graph` | T3 |
| `graph_losses.py` | confirmation and the five action losses | T3 |
| `diagnostics.py` | grid topology counts, dead directions, graph-vs-grid check | T5 |
| `train.py`, `scripts/train_data_free_latent.py` | trainer, anchor bank, CLI | T6 |

## config.py (T1)

`@dataclass class DataFreeConfig` with at least these fields (defaults in
brackets). Other modules read hyperparameters only from here.

- latent: `latent_dim [32]`
- decoder: `width [512]`, `layers [8]`, `skip_layer [4]`, `softplus_beta [20.0]`,
  `mapping_width [256]`, `mapping_layers [2]`, `base_radius R [0.5]`,
  `window_start [0.7]`, `bound_radius R_b [0.9]`, `correction_bound [0.5]`
- batch: `clusters [8]`, `codes_per_cluster [4]`, `points N [2048]`,
  `eps_min [0.05]`, `eps_max [1.0]` (chord radius of a cluster, log-uniform)
- sdf: `consistency_detach_direction [False]`, `grad_change_weight [1e-3]`,
  `grad_change_points [256]`, `grad_change_step [0.02]`
- feature size: `min_feature t [0.08]`, `barrier_floor [0.5]`
- validity: `origin_margin [0.05]`, `min_volume_fraction [0.01]`,
  `soft_tau [0.02]` (width of the soft inside indicator `sigmoid(f/tau)`)
- pose: `cov_trace_target [3*R^2/5]`
- diversity: `diversity_slope [0.1]`, `diversity_tolerance [0.25]`,
  `diversity_power [2.0]`, `phase2_start_step [20000]`,
  `descriptor_weight [1.0]`
- graph: `delta_min [0.01]`, `delta_scale [2.0]` (band half-width
  `δ = max(delta_min, delta_scale * e)`), `surface_edge_radius r_s [None]`,
  `volume_edge_radius r_v [None]`, `cos_cut [0.7]`, `facing_cos [-0.5]`,
  `max_sites [32]`, `max_fragments [8]`, `max_cavities [8]`,
  `segment_samples [8]`, `delete_size [None -> t/2]`, `delete_gap [None -> 2t]`,
  `graph_sign_error [0.05]`, `graph_full_error [0.02]`, `graph_reuse_steps [2]`
- multipliers: `tolerances: dict`, `mult_init [1.0]`, `mult_min [1e-2]`,
  `mult_max [1e4]`, `mult_rate [0.02]`, `mult_ema [0.95]`
- optimiser/trainer fields are T6's to add.

Derived helpers on the config: `r_s()` returns `surface_edge_radius` or
`2.5 * sqrt(4*pi*R^2 / N)`; `r_v()` returns `volume_edge_radius` or
`2.0 * (8 / N) ** (1/3)`.

## latent.py (T1)

- `sample_sphere(n, dim, generator=None, device=None) -> n×dim` unit vectors
  (normalised Gaussian).
- `chord(a, b)`: chord length between unit codes, broadcasting,
  `sqrt(clamp(2 - 2*cos, 0))`, safe gradient at zero.
- `slerp(a, b, t)`.
- `sample_cluster(center, k, eps, generator=None)`: `k` unit codes whose chord
  distance from `center` is at most `eps`, the first being `center` itself;
  radii spread so pairwise distances cover a range (not all at `eps`).
- `tangent_basis(z) -> (dim-1)×dim` orthonormal basis of the tangent space at
  one unit code.

## decoder.py (T1)

`class ModulatedSphereDecoder(nn.Module)`, built from a `DataFreeConfig`.

- `prepare(codes B×D) -> mod`: normalises codes to unit length, runs a mapping
  MLP, returns per-layer FiLM scale and shift tensors (computed once per code).
- `query(mod, x B×N×3) -> B×N`: the field.
- `forward(codes, x)` = `query(prepare(codes), x)`.
- Field: `f(x, z) = (R − ρ) + c(x, z)` with `ρ = sqrt(|x|² + 1e-8)` and
  `c = A · tanh(net(x, z) / A)`, `A = correction_bound` (0.35, with R = 0.5).
  There is no window by default (`use_window = False`): a fixed sphere far
  field is not the distance field of a deformed shape, so it would make the
  Eikonal and consistency losses pull every shape back to the sphere. The
  bound alone gives, for every code, `f > 0` for `ρ < R − A` (never empty,
  origin inside) and `f < 0` for `ρ > R + A < R_b` (bounded), while a true
  distance field stays representable everywhere.
  - `net`: MLP on raw `x` (no Fourier features), `layers` hidden layers of
    `width`, `Softplus(beta)`, the raw `x` concatenated back in at
    `skip_layer`. Each hidden layer is modulated:
    `h = softplus((W h + b) * (1 + scale_l) + shift_l)`.
  - Output layer and FiLM heads start with small random weights, so every code
    is close to the sphere of radius `R` but codes differ from step one (with
    identical fields the diversity loss has exactly zero gradient).

## field.py (T1)

```python
@dataclass
class FieldBatch:
    codes: Tensor    # B×D unit codes
    cluster: Tensor  # B long, cluster index of each code
    x: Tensor        # B×N×3 volume samples; identical for codes of one cluster; x[:, 0] == 0 (origin)
    f: Tensor        # B×N    field at x                (in the autograd graph)
    g: Tensor        # B×N×3  spatial gradient at x      (create_graph=True)
    xp: Tensor       # B×N×3  one-step surface guess x − f·g
    fp: Tensor       # B×N    field at xp               (in the autograd graph)
```

- `sample_points(cfg, clusters, generator, device, extra=None)`: `C×N×3`
  uniform in the box, row 0 the origin, optional `extra` points (hard
  negatives, `C×H×3`, NaN-padded) written over the last rows.
- `evaluate_field(decoder, codes, cluster, x, cfg, create_graph=True) -> FieldBatch`.
  `xp = x − f·g`; when `cfg.consistency_detach_direction` the `g` used for `xp`
  is detached (the step length `f` never is). Uses autograd for `g` (measured:
  faster than finite differences).
- `field_and_grad(decoder, mod, x, create_graph)` helper returning `(f, g)`.

## losses.py (T2)

All functions take a `FieldBatch` (and `cfg`) and return scalar tensors unless
noted. Constraint terms are non-negative and zero when satisfied.

- `eikonal(fb)`: mean `(|g| − 1)²`.
- `consistency(fb)`: mean `|fp|`. Also `consistency_error(fb)`: detached
  per-batch median of `|fp|`, the `e` used by the graph.
- `grad_change(decoder, fb, cfg, generator)`: on a random subset of
  `grad_change_points` per code, evaluate the gradient at `x + step·v` (random
  unit `v`) and penalise `1 − cos(g(x), g(x + step·v))`. Mean.
- `barrier(fb, cfg)`: where `|f| < t/2`, `relu(barrier_floor − |g|)²`; mean over
  all volume samples.
- `origin_anchor(fb, cfg)`: `relu(origin_margin − f[:, 0])²` mean.
- `soft_inside(fb, cfg)`: `sigmoid(f / soft_tau)`, B×N.
- `min_volume(fb, cfg)`: `relu(min_volume_fraction − mean soft_inside)²` mean.
- `moments(fb, cfg) -> (volume_fraction B, centroid B×3, cov B×3×3)` from
  soft-inside weights over the volume samples (uniform, so unbiased).
- `pose(fb, cfg)`: `|centroid|²` + `(trace(cov) − cov_trace_target)²` +
  sum of squared off-diagonal covariance entries; mean over codes. Axis order
  and mirror flips are deliberately not pinned.
- `descriptors(fb, cfg) -> B×K` differentiable shape statistics, with
  `DESCRIPTOR_NAMES`: volume fraction, surface area (soft count of samples with
  `|f| < band`, divided by band width, times box volume), sphericity,
  elongation and flatness (covariance eigenvalue ratios), mean inside depth
  relative to size, offset growth (volume gained by `{f > −s}` relative to
  `area·s`, a convexity proxy).
- `diversity(fb, cfg, step) -> (loss, info)`: for every pair of codes in the
  same cluster, shape distance `d = mean_n |f_i − f_j|` over the shared points
  with `|x| < R_b` (absolute difference, not relative), latent distance
  `chord`. Target `d = diversity_slope · chord`. Loss per pair:
  `relu(|d / target − 1| − diversity_tolerance) ** diversity_power`, mean over
  pairs. From `phase2_start_step` on, `d` also includes
  `descriptor_weight · mean_k |s_ik − s_jk| / scale_k` with fixed per-descriptor
  scales. `info` carries mean ratio and fraction of pairs inside the dead zone.

## multipliers.py (T2)

`class AdaptiveMultipliers`: one multiplier per named constraint.

- `update(values: dict[str, float])`: keeps an EMA of each constraint value;
  multiplies the multiplier by `(1 + mult_rate)` while the EMA exceeds that
  constraint's tolerance, by `(1 − mult_rate/2)` otherwise; clamps to
  `[mult_min, mult_max]`.
- `combine(objective, constraints: dict[str, Tensor]) -> Tensor`:
  `objective + Σ λ_k · c_k` (multipliers are plain floats, no gradient).
- `state_dict` / `load_state_dict`.

## spatial.py (T3)

- `pairs_within(a, a_shape, b, b_shape, radius, max_pairs=None) -> (ia, ib)`:
  all pairs with `a_shape == b_shape` and distance ≤ `radius`, via a hash grid
  (integer cell keys including the shape id, sort, `searchsorted` over the 27
  neighbouring cells, vectorised range expansion). No k-d tree, no dense
  `cdist` over whole shapes. When `a is b`, a flag returns each unordered pair
  once.
- `point_segment_distance(p, a, b) -> (distance, t)`.
- `connected_components(num_nodes, edge_a, edge_b) -> labels`: scatter-min label
  propagation with pointer jumping, on the GPU, over the whole batch at once.

## graph.py (T3)

`build_graph(fb_or_tensors, e, cfg, mode='full') -> ShapeGraph`, under
`torch.no_grad()`, all inputs detached. It only selects points; it never
produces gradients.

Points: per shape, the `2N` points `P = concat(x, xp)` with values
`F = concat(f, fp)` and gradients `G = concat(g, g)` (a guess reuses its parent
sample's gradient). Unit inward normal `ĝ = G / |G|`.

Quantities: `δ = max(delta_min, delta_scale·e)`. Shrunk ball radius
`ρ = max(|F| − e, 0)`.

Node classes: *inside* `F > 0`; *band* `|F| ≤ δ`; *confident outside* `F < −δ`.
Solid-graph nodes are inside ∪ band.

### Solid graph edges

1. **Surface edges**: `pairs_within` among solid nodes, radius `r_s`.
2. **Volume edges**: among *deep* nodes (inside, `ρ ≥ r_s/2`), all pairs within
   `r_v`. Deep nodes are few (hundreds per shape), so a padded dense pairwise
   per shape is fine here.
3. **Parent edges**: an inside volume sample and its own guess, when the guess
   is a solid node. Certified when `|fp| ≤ δ` (the ball claim was verified by
   the consistency check), otherwise unknown.
4. **Containment edges**: deep node `p` and any solid node `k` with
   `|p − k| ≤ ρ_p + ρ_k`. Always certified.

Edge state for families 1 and 2:

- **Certified solid** when both ends are inside and `ℓ ≤ ρ_a + ρ_b` (their
  balls overlap, so the whole segment is inside). Connectivity of the union of
  balls equals connectivity of this overlap graph, which is why no per-segment
  coverage test is needed.
- Otherwise test the cut rules; **cut** if any fires, else **unknown** with
  hidden-gap bound `w = ℓ − ρ_a − ρ_b`.

Cut rules (only evaluated for non-certified edges):

- **C1, outside ball**: some confident-outside point `o` has
  `dist(o, segment) < min(ρ_o, r_v)`. Found by `pairs_within` between edge
  midpoints and confident-outside points with a radius that covers
  `ℓ/2 + min(ρ_o, r_v)`, followed by the exact point-to-segment test.
- **C2a, endpoint crossing** (both ends band nodes): with `u` the unit vector
  from `a` to `b`, the segment leaves the solid at `a` and re-enters at `b`:
  `u·ĝ_a < −cos_cut` and `u·ĝ_b > cos_cut`.
- **C2b, mid-segment crossing**: a surfel (a guess with `|fp| ≤ δ`, normal
  `ĝ` of its parent) lies within `tol_c = r_s/2.5` of the segment, at least
  `tol_c` away from both endpoints along it, with `|ĝ·u| ≥ cos_cut`.

A nearby outside point alone never cuts (C1 needs its ball to reach the
segment). A surfel whose normal is perpendicular to the edge never cuts.

`mode='sign'` (field values not yet trustworthy): every `ρ` is zero for
certification, parent edges are unknown, and C1 uses a fixed radius `tol_c`.

### Components

- Optimistic graph: certified + unknown edges. Pessimistic: certified only.
- Labels from `connected_components`, whole batch at once.
- Main body of a shape: the component of the origin node (`x[:, 0]`) when it is
  a solid node, else the component with most nodes.

### Findings

- **Fragments**: optimistic components other than the main body
  (`kind = 0`), and pessimistic components that are not in the pessimistic main
  body but are in the optimistic one (`kind = 1`). At most `max_fragments` per
  shape, largest first. For each: size (largest `F` among its nodes), the
  closest pair of points between it and the main body (the joining segment) and
  its length.
- **Thin sites**, from surface edges between two band nodes with facing normals
  (`ĝ_a·ĝ_b < facing_cos`) and `ℓ < t`:
  - *gap site* when C2a holds (free space between two walls) **and** both ends
    are in the same optimistic component;
  - *wall site* when the mirror holds (`u·ĝ_a > cos_cut`, `u·ĝ_b < −cos_cut`:
    solid between two faces).
  Keep the `max_sites` shortest of each kind per shape. Store the midpoint and
  `ℓ`.
- **Cavities**, from the outside graph: nodes are confident-outside points,
  edges from `pairs_within` at radius `r_v`, certified when free balls overlap,
  cut when an inside ball reaches the segment or by C2b, else unknown.
  Far-field nodes are those with `|x| ≥ R_b`. An optimistic component with no
  far-field node is a cavity. At most `max_cavities` per shape; for each, the
  closest pair to the far-field component (joining segment).

### Output

```python
@dataclass
class ShapeGraph:
    # per point, B×2N
    inside: Tensor; band: Tensor; outside: Tensor       # bool
    solid_label: Tensor; solid_label_pess: Tensor       # long, −1 for non-nodes
    outside_label: Tensor                               # long, −1 for non-nodes
    main_label: Tensor                                  # B
    # fragments, F rows
    frag_shape: Tensor; frag_kind: Tensor; frag_size: Tensor; frag_gap: Tensor
    frag_a: Tensor; frag_b: Tensor                      # F×3 joining segment ends
    frag_mask: Tensor                                   # F×2N bool, member points
    # cavities, K rows
    cav_shape: Tensor; cav_a: Tensor; cav_b: Tensor; cav_mask: Tensor
    # thin sites
    gap_shape: Tensor; gap_mid: Tensor; gap_len: Tensor
    wall_shape: Tensor; wall_mid: Tensor; wall_len: Tensor
    stats: dict   # counts per edge family and state, node counts, timings in ms
```

### Efficiency target

`build_graph` for B=32, N=2048 on the RTX 5080: aim for under 60 ms, and report
the measured time and the breakdown. If the target is missed, say where the
time goes; do not silently drop a rule to hit it.

## graph_losses.py (T3)

- `confirm(decoder, mod, graph, cfg) -> ShapeGraph`: no-grad evaluation of
  `segment_samples` points along each `kind = 0` fragment's joining segment and
  each cavity's joining segment. A fragment whose segment is entirely inside is
  dropped (it was connected). A cavity whose segment is entirely outside is
  dropped. This guards the two destructive actions.
- `action_losses(decoder, mod, fb, graph, cfg) -> dict[str, Tensor]` with keys
  `bridge`, `delete`, `fill`, `widen`, `thicken` (zero tensors when nothing was
  found):
  - decide per `kind = 0` fragment: *delete* when `size < delete_size` or
    `gap > delete_gap`, else *bridge*; `kind = 1` fragments always *bridge*;
  - `bridge`: field at `segment_samples` points along the joining segment,
    `relu(−f)²` (zero margin);
  - `delete`: `relu(F)²` over the fragment's member points, using the values
    already in `fb` (no re-evaluation);
  - `fill`: `relu(−F)²` over the cavity's member points, from `fb`;
  - `widen`: field at gap-site midpoints, only where it is negative there,
    `relu(f + t/2)²`;
  - `thicken`: field at wall-site midpoints, only where it is positive there,
    `relu(t/2 − f)²`.
- `hard_negatives(graph, fb, cfg, per_shape) -> B×H×3` (NaN-padded): points of
  fragments, cavities and thin sites, for the trainer's anchor cache.

## diagnostics.py (T5)

- `grid_topology(field_fn, resolution=64) -> dict`: evaluates a field on a
  grid over the box and returns pieces (components of `f > 0`), cavities
  (components of `f ≤ 0` not touching the box boundary), and handles from the
  Euler number (`handles = pieces + cavities − euler`).
- `graph_topology(graph) -> per-shape pieces and cavities`, and
  `compare_graph_to_grid(...)`: agreement rate between the two on a batch (the
  plan's "check before using it in a loss").
- `dead_directions(decoder, code, cfg, ...) -> dict`: plan section 7. Surface
  points are guesses with small `|fp|`; outward normal displacement per unit
  code step along each of the `D−1` tangent directions is `df / |g|`; remove by
  least squares the span of translation, rotation and uniform scale; smooth
  over neighbours within a `planner_resolution` radius; singular values (RMS
  displacement per unit chord). Returns live-direction count (motion over
  `cluster_radius` exceeding the consistency error), wobble share (energy
  removed by smoothing) and the pose share. `subspace_stability(...)` compares
  the live subspaces of two nearby codes.

## train.py and CLI (T6)

- Anchor bank: a fixed set of unit codes with a per-anchor cache of
  hard-negative points. Each step draws cluster centres partly from the bank,
  partly fresh.
- Step: sample clusters and shared points → `evaluate_field` → SDF/validity/
  pose/barrier/diversity terms → consistency error `e` picks the graph mode
  (`off` above `graph_sign_error`, `sign` above `graph_full_error`, else
  `full`) → `build_graph` (no grad) → `confirm` → `action_losses` →
  `AdaptiveMultipliers.combine` → optimiser step. The graph is reused for
  `graph_reuse_steps` optimiser steps on the same codes and points.
- Diversity is the objective; every other term is a constraint with a
  tolerance.
- Periodic diagnostics: grid topology (pieces, cavities, handles) on anchor
  codes, graph-vs-grid agreement, dead directions; TensorBoard scalars;
  checkpoints with full resume state.
