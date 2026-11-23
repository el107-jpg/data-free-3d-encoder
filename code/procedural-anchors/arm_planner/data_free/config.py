"""Hyperparameters of the data-free latent space. Other modules read only from here."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple


def default_tolerances() -> Dict[str, float]:
    """Tolerance of each named constraint (the value below which it counts as met)."""
    return {
        "eikonal": 1e-2,       # mean (|g| - 1)^2
        "consistency": 5e-3,   # mean |f(x - f g)|
        "grad_change": 2e-2,   # mean 1 - cos between nearby gradients
        "barrier": 1e-4,       # mean relu(floor - |g|)^2 over the near-surface samples
        "origin": 1e-6,        # mean relu(margin - f(0))^2
        "volume": 1e-6,        # mean relu(min fraction - soft volume)^2
        # centroid, covariance trace and off-diagonals. The soft moments of an
        # exactly centred sphere of the right size measure 1.0e-3 at N = 2048
        # (sampling noise, mostly of the centroid; 2.8e-4 at N = 8192), so the
        # tolerance must sit above that or the multiplier runs to mult_max.
        "pose": 2e-3,
        # The trainer uses the three unbiased split-sample terms of
        # losses.pose_terms instead of "pose" (kept for the biased sum).
        # |centroid|^2: centroid within 0.016. Its sampling noise is 1e-3 per
        # shape at N = 2048, so 1e-4 kept the multiplier pinned on noise.
        "pose_centroid": 2.5e-4,
        "pose_scale": 5e-5,     # (trace - target)^2: trace within about 5%
        "pose_axes": 1e-4,      # sum of squared off-diagonal covariance entries
        "bridge": 1e-5,        # relu(-f)^2 on joining segments
        "delete": 1e-5,        # relu(f)^2 on fragment points
        "fill": 1e-5,          # relu(-f)^2 on cavity points
        # A single thin site in one shape of a batch of 32 reads up to about 5e-5, and such
        # sites come and go. At 1e-5 that kept the multiplier rising to its cap (measured:
        # 8000 within 300 updates), which then distorted every shape. 1e-4 tolerates an
        # isolated site and reacts when thin features persist across several shapes.
        "widen": 1e-4,         # relu(f + t/2)^2 at gap sites
        "thicken": 1e-4,       # relu(t/2 - f)^2 at wall sites
    }


ACTION_CONSTRAINTS = ("bridge", "delete", "fill", "widen", "thicken")


def default_mult_ranges() -> Dict[str, Tuple[float, float]]:
    """Per-constraint multiplier range; names not listed use [mult_min, mult_max].

    The action losses and the barrier need large multipliers to remove a
    detached blob or a thin wall once it exists (measured: with a cap of 200
    blobs persisted for thousands of steps, with 1e4 they were gone in 1500).
    The pose terms are unbiased but noisy, so their multipliers stay small:
    the gradient of pose_centroid is almost pure sampling noise at N = 2048.
    """
    ranges: Dict[str, Tuple[float, float]] = {name: (1.0, 1e4) for name in ACTION_CONSTRAINTS}
    ranges["barrier"] = (1.0, 1e4)
    # Thicken and widen act at single midpoints; at weights in the thousands they bend the
    # field around the site instead of moving the wall, so they are capped lower.
    ranges["widen"] = (1.0, 1e3)
    ranges["thicken"] = (1.0, 1e3)
    ranges["pose_centroid"] = (1.0, 50.0)
    # Light anchors only: pose is removed from the diversity distance itself.
    ranges["pose_scale"] = (1.0, 50.0)
    # The axes term only has something to hold once shapes are elongated. With elongation
    # around 1.4 a cap of 50 left the axes loose (mean off-diagonal correlation 0.17) and
    # rotation at 11% of the surface motion, so it gets a higher cap. Its sampling noise is
    # small (about 7e-5 per shape), unlike the centroid term's.
    ranges["pose_axes"] = (1.0, 500.0)
    return ranges


def default_mult_rates() -> Dict[str, float]:
    """Per-constraint multiplier rate; names not listed use mult_rate.

    0.05 takes a multiplier from 1 to 1e3 in about 140 violated steps.
    """
    rates = {name: 0.05 for name in ACTION_CONSTRAINTS}
    rates["barrier"] = 0.05
    return rates


@dataclass
class DataFreeConfig:
    # latent
    latent_dim: int = 32

    # decoder
    width: int = 512
    layers: int = 8
    skip_layer: int = 4
    softplus_beta: float = 20.0
    # activation_leak > 0 makes the main network's activation leak * x + (1 - leak) * softplus(x):
    # still smooth, slope 1 when a unit is on, but slope `leak` (not zero) when it is off, so a unit
    # pushed far negative keeps a gradient and can come back. Measured without it: 99 of the 512
    # units of the last hidden layer dead by update 1,000 and 405 by update 7,000. 0 = plain softplus.
    activation_leak: float = 0.0
    # layer_norm_layers = n layer-normalises the last n hidden layers of the main network (8 = all,
    # 1 = only the one that feeds the output, 0 = none). At every point the layer's 512 values are
    # brought to mean 0 and variance 1 across the units before the code's scale and shift, so about
    # half the units are on at any point. Normalising across units, not across points, keeps the
    # field a plain function of code and position, which the derivative terms need.
    layer_norm_layers: int = 0
    # fourier_bands = n adds sin and cos of pi 2^k x (k < n) to the coordinates the main network sees,
    # at its input and at the halfway skip. A smooth network on raw coordinates learns broad shape
    # first and fine detail very slowly; these let it draw detail down to a period of 2 / 2^(n-1)
    # (3 bands: 0.5; 5 bands: 0.125). The price is that ripple and specks become just as easy.
    # 0 = raw coordinates, as the design note asks.
    fourier_bands: int = 0
    mapping_width: int = 256
    mapping_layers: int = 2
    base_radius: float = 0.5        # R, radius of the initial sphere
    correction_bound: float = 0.35  # A in f = (R - rho) + A tanh(net / A)
    # How far the field may fall BELOW the sphere's field (None = correction_bound, the symmetric
    # form). With the symmetric form f >= (R - A) - rho, so every shape contains the ball of
    # radius R - A around the origin: plates thinner than that, and shapes whose centroid lies
    # outside them, cannot exist. A value of R + (R + A), 1.35 at the defaults, lets the field be
    # as negative as any shape inside radius R + A could need and removes that limit; the outer
    # limit (shape inside radius R + A) is unchanged. Set origin_margin to a large negative
    # number with it, or the origin term still asks for a filled centre.
    correction_bound_inner: Optional[float] = None
    bound_radius: float = 0.9       # R_b > R + A: guaranteed free space from here on
    output_init_std: float = 2e-4   # std of the output layer weights at init
    film_init_std: float = 0.1      # std across codes of each FiLM scale/shift at init
    use_window: bool = False        # True: fade the correction out between window_start and R_b
    window_start: float = 0.7       # (use_window only) correction window is 1 up to here
    cull_far_field: bool = True     # (use_window only) skip the network where the window is zero

    # batch
    clusters: int = 8
    codes_per_cluster: int = 4
    points: int = 2048              # N, volume samples per code
    eps_min: float = 0.05           # chord radius of a cluster, log-uniform
    eps_max: float = 1.0

    # sdf
    consistency_detach_direction: bool = False
    # Consistency is the binding limit on variety (its tolerance sets how much the shapes may
    # differ), so two ways of easing it:
    # * A schedule. consistency_tolerance_start (None = no schedule) is the tolerance for the
    #   first consistency_hold_steps updates; it then falls geometrically to
    #   tolerances["consistency"] over consistency_tighten_steps updates. Keep the start at or
    #   below about 0.03: the graph is fully on only while the median miss is under
    #   graph_full_error (0.02), and the mean runs about 1.5 times the median.
    # * consistency_fixed_surface. The term |f(x - f g)| has two gradient routes: through the
    #   field where the step lands (on the surface, so it reshapes the object) and through the
    #   field at the starting point (so the step lands correctly). True blocks the first: the
    #   landing value is computed with the network's weights held fixed, so the term can only
    #   adjust the field around the surface to agree with it. consistency_min_depth then leaves
    #   out starting points closer to the surface than this, which would otherwise still move it.
    # * A ramp on the weights. field_ramp_steps > 0 raises the upper limits of the eikonal and
    #   consistency multipliers geometrically from field_ramp_start to their configured values
    #   over that many updates (the lower limits drop with them), so variety can develop before
    #   the two field-accuracy terms are back at full strength.
    field_ramp_steps: int = 0
    field_ramp_start: float = 0.01
    # * Even use of the code. code_isotropy_weight > 0 adds losses.code_isotropy: a step of the same
    #   size in any code direction should change the scales and shifts handed to the main network
    #   by the same amount (variance of the log change over code_isotropy_directions random
    #   directions per code). It acts on the side network only and costs next to nothing. It does
    #   not by itself make the main network turn those changes into shape changes.
    code_isotropy_weight: float = 0.0
    code_isotropy_directions: int = 16
    consistency_tolerance_start: Optional[float] = None
    consistency_hold_steps: int = 0
    consistency_tighten_steps: int = 0
    consistency_fixed_surface: bool = False
    consistency_min_depth: float = 0.0
    # consistency_surface_scale: a dial between the two. The gradient of the landing route is
    # multiplied by this (1 = ordinary consistency, 0 = the same as consistency_fixed_surface).
    # Cutting the route completely made training grow shapes and then fall back to the sphere;
    # a fraction keeps a restoring link between the field and the surface. Same cost as
    # ordinary consistency. Ignored when field_moves_surface is on.
    consistency_surface_scale: float = 1.0
    # A tidy-up stage that repairs the field without reshaping the objects:
    # * variety_weight scales the variety objective (0 switches the push off);
    # * shape_hold_weight > 0 keeps a frozen copy of the decoder as it was when the stage began
    #   and penalises any change of the field within shape_hold_band of that copy's surfaces
    #   (mean absolute difference there), so the shapes stay where they are while the
    #   field-accuracy terms (with consistency_fixed_surface and consistency_min_depth) work on
    #   the field around them. Costs one extra forward pass without gradients.
    # * shape_hold_newton > 0 pins the surface itself instead of the band: the sample points are
    #   slid onto the frozen copy's surface with that many Newton steps and the term is the mean
    #   |f| of the trained decoder at those points. The field NEXT TO the surface is then free to
    #   change, which the band version forbids (it holds the old, possibly wrong, values there).
    #   Costs that many forward and backward passes of the frozen copy plus one of the decoder.
    # * field_only leaves only the eikonal and consistency terms (and the hold) in the loss; the
    #   pose, barrier and graph terms are still measured and logged but push nothing.
    variety_weight: float = 1.0
    # variety_surface_band > 0 makes the variety push compare two shapes only at the sample points
    # within that distance of either shape's surface. There the difference of the two fields is the
    # distance from one surface to the other, so the pair's distance becomes a surface (Chamfer)
    # distance: a thin limb counts by how far its surface is from the other shape, not by its volume,
    # and a difference of the fields far from both surfaces counts for nothing. 0 = the whole ball.
    # Costs nothing extra (the fields at shared points are already there); fewer points per pair.
    variety_surface_band: float = 0.0
    shape_hold_weight: float = 0.0
    shape_hold_band: float = 0.04
    shape_hold_newton: int = 0
    shape_hold_converged: float = 2e-3
    field_only: bool = False
    # field_moves_surface: with the landing route blocked, a push on the field at a point far
    # from the surface no longer reaches the shape (consistency just pulls the value back), and
    # the fixed-surface run barely diversified or reproduced its anchors. True makes every push on
    # the field value at a sample point (from the neighbour term and from the anchor pull) also
    # act on the field at that point's surface guess, i.e. on the piece of surface the point
    # belongs to. Values are unchanged; only where the gradient lands changes. Costs one extra
    # backward pass through the network.
    field_moves_surface: bool = False
    grad_change_weight: float = 1e-3
    grad_change_points: int = 256
    grad_change_step: float = 0.02

    # feature size
    min_feature: float = 0.08       # t
    barrier_floor: float = 0.5

    # validity
    origin_margin: float = 0.05
    min_volume_fraction: float = 0.01
    soft_tau: float = 0.02          # soft inside indicator sigmoid(f / tau)

    # pose
    # None -> covariance trace of the ball of radius R as the soft indicator
    # sigmoid(f / soft_tau) measures it (3 R^2 / 5 times 1.037 at the defaults),
    # so the initial sphere has zero scale error.
    cov_trace_target: Optional[float] = None
    # Where a shape sits and how big it is are measured with the power ``pose_power`` (p) of the
    # distance from the origin, weighted by the soft inside indicator w:
    #   centre = the point c that minimises  sum w |x - c|^p        (must be the origin),
    #   size   = (sum w |x|^p / sum w)^(1/p)                         (must equal that of the ball of radius R).
    # p = 2 is the mass centre and the covariance trace (the default; a config saved without this
    # field loads as 2). A larger p weighs the far parts of a shape more, and p -> infinity is the
    # centre and radius of the smallest enclosing ball: a ball with a long rod is then posed with a
    # smaller ball instead of the rod leaving the decoder's outer radius.
    # Orientation is unchanged (principal axes of the mass along the coordinate axes).
    # pose_moment_target: None -> the p-th moment of the soft ball of radius R, so the initial sphere
    # has zero size error.
    #
    # MEASURED 12 Nov 2026: p = 6 does NOT train with the anchor recipe. Three runs (P1, P2, X2) ended
    # with a quarter to a half of the random codes decoding to an empty shape, with either size-weight
    # ceiling, while the same recipe and anchors with p = 2 stayed clean (X1). With p = 6 "size" means
    # reach, and a shape that is too small can only grow by pushing thin parts further out, which the
    # decoder does last; with p = 2 it fattens, which is easy. Shapes were at 0.69 of the target size by
    # update 100 and never above about 0.8 (0.93 with the ceiling at 500); the size term then keeps
    # pushing on the field near the far tips, the consistency error rises past the level at which the
    # graph terms switch off, and shapes fragment and empty. To admit far-reaching ANCHOR shapes keep
    # p = 2 and scale those anchors down to fit instead (build_data_free_procedural_anchors.py
    # --fit-reach, with anchor_exempt_size).
    pose_power: float = 2.0
    pose_moment_target: Optional[float] = None

    # diversity
    diversity_slope: float = 0.1
    diversity_tolerance: float = 0.25
    diversity_power: float = 2.0
    # Count only the part of a pair's field difference that no small translation, rotation or
    # resize explains (see losses._remove_pose). Without it, turning or resizing a shape is the
    # cheapest diversity, and the moment-based pose terms cannot pin a nearly round shape.
    diversity_pose_invariant: bool = True
    # "slope": shape distance must stay within a band around diversity_slope x latent distance.
    # "ordinal": diversity_slope is only a floor, and what is enforced above it is the ORDER of
    # distances (a code nearer in latent space is nearer in shape), with no curve to follow.
    # ordinal_margin: required relative gap in shape distance as a fraction of the relative gap
    # in latent distance (0 = bare ordering). ordinal_tie: latent distances closer than this
    # fraction are treated as ties and not compared.
    # "repel": no floor or ordering, a logarithmic push on the latent-distance-weighted total.
    # "nearest": each shape is pushed away from its nearest neighbours in the batch (a soft
    # minimum with exponent nearest_power); intended for global_codes, where the batch is
    # clusters x codes_per_cluster independent random codes sharing one set of sample points.
    global_codes: bool = False
    nearest_power: float = 4.0
    # Pooled neighbour search (global_codes with diversity_mode "nearest" only): pool_size further
    # random codes are evaluated without gradients at pool_points of the batch's sample points,
    # and each batch shape is pushed away from its nearest_k most similar shapes among the batch
    # and the pool. 0 keeps the search inside the batch.
    # batch_neighbours (needs accumulate > 1): the passes of an update share their first
    # pool_points sample points, all of the update's codes are evaluated there once without
    # gradients, and every shape is compared with all the others of the update. Summed over the
    # passes this is the exact gradient of the nearest-neighbour loss over the whole update.
    # pool_size then adds an extra crowd evaluated once per update.
    batch_neighbours: bool = False
    pool_size: int = 0
    pool_points: int = 1024
    nearest_k: int = 8
    # Weak-direction pairs (batch_neighbours mode only). The decoder reads the same few fixed
    # directions of code space at every code and ignores the rest; two random codes differ in
    # all of them, so an unused direction adds next to nothing to their distance and gets next
    # to no push. Every weak_refresh_every updates the trainer measures how much each direction
    # of code space changes the field (weak_probe_codes codes x weak_probe_points points, no
    # gradients). Directions whose field change per unit chord (RMS) is at or below weak_floor
    # count as unused. In each pass weak_pairs of the codes are then not random: each is a
    # partner of another code of the pass, with exactly the same used part and its unused part
    # turned so that the two sit weak_chord apart. A pair is pushed apart by
    # -log(distance / (diversity_slope * chord) + weak_pair_eps), pose removed, weighted by
    # weak_pair_weight per pair (a pair counts like one code of the neighbour term). Partners
    # stay out of the neighbour search. 0 turns it off.
    weak_pairs: int = 0
    weak_chord: float = 0.25
    weak_pair_weight: float = 1.0
    weak_pair_eps: float = 0.05
    weak_floor: float = 0.04
    weak_refresh_every: int = 500
    weak_probe_codes: int = 32
    weak_probe_points: int = 1024
    # Frame form of the same idea (used instead of weak_pairs when weak_frame_passes > 0). A
    # pair only has to end up different, and the cheapest way is to make an unused direction
    # move the shape along a pattern the decoder already has: the used axes turn into the unused
    # set and no new way of varying appears (measured: 54% of the usage moved, the count of
    # directions did not). In the first weak_frame_passes passes of an update the first code
    # therefore gets one partner along every used direction and weak_frame_new partners turned in
    # the unused ones, all weak_chord away, and the term is
    # -log det(G + weak_frame_sigma^2 I) / partners, with G the Gram matrix of the partners'
    # pose-corrected field changes per unit chord. A change that copies an existing pattern adds
    # no volume; only a new pattern does. weak_frame_weight scales the term (per update).
    # weak_frame_full: the frame is the complete tangent frame at the first code (a partner along
    # every one of the latent_dim - 1 directions, which fills the pass). The log-volume of a
    # complete frame does not depend on the choice of perpendicular directions, so no usage is
    # measured and weak_floor, weak_frame_new and weak_refresh_every are not used. It rewards
    # every direction for a change of its own and so evens out the directions that exist; it
    # did not revive directions a trained model had stopped reading, so it is meant for runs
    # where something else (anchor objects) keeps every direction in use.
    weak_frame_passes: int = 0
    weak_frame_new: int = 6
    weak_frame_weight: float = 1.0
    weak_frame_sigma: float = 0.02
    weak_frame_full: bool = False
    # Anchor objects (batch_neighbours mode only). anchor_path is a file written by
    # scripts/build_data_free_anchors.py: distance grids of real objects, already in the trainer's
    # pose convention and inside the decoder's hard limits. Each object owns one code; the codes
    # start mutually perpendicular (a random orthonormal frame, then its negatives when there are
    # more objects than latent dimensions), so reproducing the objects needs every direction of
    # the code. In each pass anchor_codes_per_pass of the random codes are replaced by anchor
    # codes (a different subset every update), and the decoder's field at those codes is pulled
    # onto the object's field at the points the pass already samples:
    #   anchor_weight * ( mean |f - target| inside the bound radius
    #                     + anchor_surface_weight * mean |target(surface guess)| ).
    # Anchor codes stay out of the neighbour search. They are fixed for anchor_freeze_steps
    # updates; after that they are trained with a learning rate that ramps linearly to
    # anchor_embed_lr over anchor_ramp_steps (0 keeps them fixed for good) and are put back on
    # the unit sphere after every step. anchor_count uses only the first objects of the file
    # (they are stored in selection order, most distinct first); 0 uses all.
    anchor_path: str = ""
    anchor_count: int = 0
    anchor_codes_per_pass: int = 4
    anchor_weight: float = 100.0
    anchor_surface_weight: float = 1.0
    anchor_surface_band: float = 0.1
    anchor_embed_lr: float = 0.0
    anchor_freeze_steps: int = 500
    anchor_ramp_steps: int = 1000
    # anchor_radius > 0: an anchor acts on a neighbourhood of its code, not on the one code. Each
    # time an anchor is drawn it is met at a random code up to 2 * anchor_radius (chord) away,
    # and the pull is weighted by exp(-(distance / anchor_radius)^2). The decoder then cannot
    # reproduce an object in an isolated spike at its code and ignore it everywhere else: the
    # shape has to change gradually on the way out. 0 pulls at the anchor's code only.
    anchor_radius: float = 0.0
    # anchor_exempt_field: leave the anchor rows out of the eikonal and consistency constraints.
    # They are tied to exact distance values by the pull itself, and they carry about twice the
    # consistency error of the decoder's own shapes (measured 0.009-0.010 against 0.0044, with a
    # tolerance of 0.005), which holds the batch-mean constraint at its limit.
    anchor_exempt_field: bool = False
    # anchor_exempt_size: leave the anchor rows out of the size term (pose_scale). For anchor files in
    # which far-reaching shapes were scaled down to fit inside the outer radius: those anchors are
    # smaller than the size target on purpose.
    anchor_exempt_size: bool = False
    diversity_mode: str = "slope"
    ordinal_weight: float = 1.0
    ordinal_margin: float = 0.25
    ordinal_tie: float = 0.1
    # Weight of losses.spread in the objective: different latent directions must change the
    # shape in different ways. 0 turns it off. spread_floor is the pattern size (as a fraction
    # of diversity_slope) below which a direction counts as not moving the shape.
    spread_weight: float = 0.0
    spread_floor: float = 0.1
    # Weight of losses.statistic_repulsion in the objective: each group of shape statistics
    # (proportions, thickness, compactness) is pushed apart on its own, with no target. 0 = off.
    # When used, keep the statistics out of the pair distance (phase 2 off), so that only the
    # field distance carries the floor and the ordering.
    stat_repulsion_weight: float = 0.0
    stat_repulsion_eps: float = 1e-3
    # Weight of losses.local_chamfer in the objective: surfaces of codes in a cluster are pushed
    # apart where they are locally far from each other (nearest-point search between the existing
    # surface samples, distance to the tangent plane, thresholded at the surface's own sampling
    # noise). 0 = off. No network evaluations.
    local_chamfer_weight: float = 0.0
    local_chamfer_queries: int = 512
    local_chamfer_quantile: float = 0.75
    local_chamfer_leak: float = 0.1
    phase2_start_step: int = 20000
    # The field distance is of order diversity_slope * chord (0.005 to 0.1) and
    # scaled descriptor differences are of order 1, so this weight is small.
    descriptor_weight: float = 0.02

    # graph
    delta_min: float = 0.01
    delta_scale: float = 2.0
    surface_edge_radius: Optional[float] = None  # None -> see r_s()
    volume_edge_radius: Optional[float] = None   # None -> see r_v()
    cos_cut: float = 0.7
    facing_cos: float = -0.5
    max_sites: int = 32
    max_fragments: int = 8
    max_cavities: int = 8
    segment_samples: int = 8
    delete_size: Optional[float] = None  # None -> t / 2
    delete_gap: Optional[float] = None   # None -> 2 t
    graph_sign_error: float = 0.05
    graph_full_error: float = 0.02
    graph_reuse_steps: int = 2
    # Stricter-than-spec rules of graph.py (its DEVIATION knobs), same defaults.
    graph_one_sided_cos: float = 0.3
    graph_one_sided_cut: bool = True
    graph_hidden_gap_drop: bool = True
    graph_kind1_core: Optional[float] = None       # None -> r_s()
    graph_rescue_degree: int = 4
    graph_rescue_cos: float = 0.05
    graph_gap_resolution: Optional[float] = None   # None -> t / 2
    graph_debug: bool = False

    # multipliers
    tolerances: Dict[str, float] = field(default_factory=default_tolerances)
    mult_init: float = 1.0
    # Range of every multiplier not listed in mult_ranges. The split-sample pose terms are unbiased but
    # noisy: at N = 2048 the gradient of pose_centroid is almost pure sampling
    # noise once the centroid is within tolerance (cosine -0.04 between two
    # point samples of the same codes), so a multiplier in the thousands
    # (measured: 3680 with the old cap of 1e4) makes that noise 14 times the
    # diversity gradient. 200 keeps it below the objective. The floor of 1
    # (was 1e-2) keeps a satisfied constraint from needing hundreds of steps
    # to wake up again.
    mult_min: float = 1.0
    mult_max: float = 200.0
    mult_rate: float = 0.02
    mult_ema: float = 0.95
    mult_ranges: Dict[str, Tuple[float, float]] = field(default_factory=default_mult_ranges)
    mult_rates: Dict[str, float] = field(default_factory=default_mult_rates)

    # optimiser (AdamW, linear warm-up then cosine decay to a floor; no mixed precision)
    steps: int = 50000              # total optimiser steps (the cosine horizon)
    lr: float = 3e-4
    lr_warmup_steps: int = 500
    lr_floor_fraction: float = 0.05  # final lr as a fraction of lr
    # Step at which the cosine decay reaches the floor; None means the end of the run (steps).
    # Set it below ``steps`` to hold the rate at the floor for the rest of a run, e.g. when
    # continuing a finished model at a constant rate.
    lr_decay_steps: Optional[int] = None
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    grad_clip: float = 1.0          # gradient-norm clip; <= 0 disables it
    # Sub-batches whose gradients are averaged into one optimiser step (clusters x
    # codes_per_cluster shapes each). 1 keeps the original one-batch step with graph reuse.
    accumulate: int = 1

    # anchor bank
    bank_size: int = 4096           # fixed unit codes drawn once
    bank_fraction: float = 0.5      # share of cluster centres taken from the bank
    hard_negatives: int = 64        # cached hard-negative points per anchor

    # trainer
    seed: int = 0
    log_every: int = 50             # console line and TensorBoard scalars
    diag_every: int = 500           # periodic diagnostics; <= 0 disables them
    ckpt_every: int = 2000          # checkpoints; <= 0 keeps only the final one
    probe_codes: int = 16           # first anchors of the bank, used by the diagnostics
    diag_resolution: int = 48       # voxel grid of the topology diagnostics
    dead_direction_probes: int = 4  # probe codes given the dead-direction diagnostic
    dead_direction_points: int = 4096
    dead_direction_neighbour: float = 0.1  # chord to the code used for subspace stability
    diag_pose_points: int = 262144  # samples for the probes' measured pose

    def __post_init__(self) -> None:
        # Derived defaults are resolved once, from the values given at
        # construction. Set them explicitly if base_radius or min_feature is
        # changed afterwards.
        if self.cov_trace_target is None:
            q = (math.pi * self.soft_tau / self.base_radius) ** 2
            self.cov_trace_target = (
                3.0 * self.base_radius ** 2 / 5.0 * (1.0 + 10.0 * q / 3.0 + 7.0 * q * q / 3.0) / (1.0 + q)
            )
        if self.delete_size is None:
            self.delete_size = self.min_feature / 2.0
        if self.delete_gap is None:
            self.delete_gap = 2.0 * self.min_feature
        if not float(self.pose_power) >= 2.0:
            raise ValueError("pose_power must be at least 2")
        if not 0.0 < self.window_start < self.bound_radius <= 1.0:
            raise ValueError("need 0 < window_start < bound_radius <= 1")
        if not 0.0 < self.correction_bound < self.base_radius:
            # f(0) >= R - A > 0: the origin is inside every shape (unless correction_bound_inner
            # is set, which widens the inward side only)
            raise ValueError("need 0 < correction_bound < base_radius")
        if self.correction_bound_inner is not None and not self.correction_bound_inner > 0.0:
            raise ValueError("need correction_bound_inner > 0")
        if not self.base_radius + self.correction_bound < self.bound_radius:
            # f < 0 for rho > R + A: everything beyond R_b is free space
            raise ValueError("need base_radius + correction_bound < bound_radius")

    # Short names used in the design text.
    @property
    def R(self) -> float:
        return self.base_radius

    @property
    def R_b(self) -> float:
        return self.bound_radius

    @property
    def t(self) -> float:
        return self.min_feature

    @property
    def N(self) -> int:
        return self.points

    @property
    def batch_size(self) -> int:
        return self.clusters * self.codes_per_cluster

    def r_s(self) -> float:
        """Surface edge radius: 2.5 sample spacings on the sphere of radius R."""
        if self.surface_edge_radius is not None:
            return float(self.surface_edge_radius)
        return 2.5 * math.sqrt(4.0 * math.pi * self.base_radius ** 2 / self.points)

    def r_v(self) -> float:
        """Volume edge radius: 2 sample spacings in the box."""
        if self.volume_edge_radius is not None:
            return float(self.volume_edge_radius)
        return 2.0 * (8.0 / self.points) ** (1.0 / 3.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataFreeConfig":
        """Build from a dict, ignoring keys this version does not know."""
        known = {f.name for f in fields(cls)}
        if "pose_power" not in data:
            # saved before the extent-based pose rule existed: it was trained with the mass-based one
            data = {**data, "pose_power": 2.0}
        return cls(**{k: v for k, v in data.items() if k in known})
