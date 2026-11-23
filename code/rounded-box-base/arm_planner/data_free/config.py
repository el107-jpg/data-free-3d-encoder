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
    # weight_norm stores each unit's incoming weights in the main network (hidden layers and the
    # output layer) as a direction and a size, w = g v / |v|, as DeepSDF does. Same function at the
    # start; what changes is the pace of learning. Without it the output layer, whose weights are
    # about 0.004, changes by about 15% of itself per update while a hidden layer changes by about
    # 1.5%, and nearly all of the variety gradient acts on the output layer. With it every unit's
    # direction turns at the same pace whatever the size of its weights.
    weight_norm: bool = False
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
    # sphere_base = False drops the fixed sphere from the formula: the main network's own output
    # is the field (no bound on it), blended into the sphere's field R - rho between window_start
    # and the bounding radius so that nothing can reach beyond the bounding radius. With the
    # sphere as a fixed base, a surface that does not face away from the centre has to cancel the
    # sphere's slope first (the correction needs slope sqrt(2 - 2 cos) there: 0 facing outward,
    # 1.4 side-on, 2 facing the centre), which favours star-shaped blobs. Without it the network
    # must START as a sphere instead: fit one with analysis_scripts/sphere_fit.py and pass the
    # weights as init_main_from (the main network's weights of a new run are loaded from there;
    # the side network keeps its usual start, and film_init_std should be about 0.05 times the
    # usual value for codes to start as close together as they do with the base).
    sphere_base: bool = True
    # How a decoder without the sphere base is kept inside the bounding radius. "blend" (the first
    # version) blends the network into R - rho between window_start and the bounding radius; that
    # pins the field there to a sphere of radius R, which is wrong for every other shape and
    # leaves the slope term unsatisfiable (measured: slope about 0.5 between a shrunken surface
    # and the pinned ring). "min" takes the smooth minimum of the network's output and the
    # bounding ball's field R_b - rho: the shape cut off by the ball, with nothing pinned inside.
    # The network is then evaluated at every point (no skipping beyond the bounding radius).
    outer_limit: str = "blend"
    init_main_from: Optional[str] = None
    # The fixed part of the field. "sphere": R - rho (every code shares it). "rounded_box": a
    # rounded box set by the code, r - (|max(q, 0)| + min(max_i q_i, 0)) with q = |x| - b, an
    # exact signed distance (slope 1 everywhere). Half-extents b (ordered b1 >= b2 >= b3 >= 0) and
    # rounding radius r come from a small head on the side network, initialised at zero so that
    # every code starts at b = 0, r = R: exactly the sphere. b = (L, 0, 0) is a capsule (rod),
    # b = (a, a, 0) a slab of thickness 2r, a small r a sharp-edged box. The correction on top is
    # unchanged, so the outer guarantee becomes "shape inside the base grown by A".
    base_shape: str = "sphere"
    # Largest distance of the base from the origin, |b| + r (its corner). None = bound_radius -
    # correction_bound, the most that keeps the shape inside the bounding radius (0.55 at the
    # defaults): a rod can be 2 x 0.55 long, a square plate about 2 x 0.39 wide, a cube 2 x 0.32.
    base_max_reach: Optional[float] = None
    base_round_min: float = 0.05    # smallest rounding radius r (half-thickness of the thinnest rod/sheet)
    # The base may leave the sphere gradually: b and r - R are scaled by min(1, step / ramp).
    base_ramp_steps: int = 0
    # The head's output is multiplied by this. Under Adam this sets how fast the base moves
    # relative to the rest of the network (its learning-rate factor, without a separate group).
    base_speed: float = 1.0
    # The main network also sees, at every point, the base's value base(x) and the nearest point
    # of the base's core box clamp(x, -b, b) (4 numbers, appended after the coordinates). The
    # residual can then work relative to the base surface. Their input weights start at zero.
    base_features: bool = True
    # With a value k, the inward bound of the correction is k r for each code (instead of
    # correction_bound_inner): f >= base - k r, so the core box grown by (1 - k) r is always
    # inside the shape (one piece, never cut through; to within 0.007, the most an extent can dip
    # below zero). None keeps correction_bound_inner.
    base_inner_fraction: Optional[float] = None
    # Form of the learned head's output (base_map "learned").
    # "squashed" (B1): extents through a soft ramp of temperature 0.01, ordered, radius through a
    # sigmoid; the ramp went flat after a few updates' drift and the head never recovered.
    # "linear": the head's output IS the deviation from the sphere: b_i = u_i, r = R + u_r, each axis on
    # its own. Only two hard limits: r keeps above base_round_min, and the corner |b| + r is capped at
    # the reach (the outer guarantee). Negative extents (a lens) are allowed but cost
    # base_negative_weight * mean(relu(-b)^2); below -0.1 a soft floor keeps the base from vanishing.
    base_head_form: str = "squashed"
    # Where the learned head reads from. "shared": a linear head on the side network's features (shared
    # with FiLM, so the head's gradient also trains the side network). "shared_mlp": the same features
    # through one more nonlinear layer, features -> base_net_width (tanh) -> 4. "separate": its own small
    # network on the unit code, code -> base_net_width (tanh, zero-centred) -> 4; it shares nothing with
    # the side network or FiLM. In every form the last layer starts at zero (the sphere).
    base_head_net: str = "shared"
    base_net_width: int = 64
    # Also give the main network the base parameters themselves, (b1, b2, b3, r - R) at every point,
    # appended to the base features (so at layer 0 and again at the skip layer), weights starting at 0.
    base_param_inputs: bool = False
    base_negative_weight: float = 100.0
    # Deviation budget (base_head_form "linear"): the mean over codes of |b|^2 + (r - R)^2 is an
    # adaptive constraint named base_deviation with tolerance budget^2, the budget growing linearly
    # from base_budget_start to base_budget over base_budget_steps updates. Its weight starts at
    # base_budget_mult_init (stiff: the base stays at the sphere) and, like every constraint weight,
    # decays while the deviation is under the budget and grows while it is over. 0 = no budget.
    base_budget: float = 0.0
    base_budget_start: float = 0.01
    base_budget_steps: int = 500
    base_budget_mult_init: float = 1000.0
    # Learning-rate factor of the learned base head's own weights (base_head / base_head_mlp / base_net; the
    # side network it may share with FiLM keeps the full rate). Below 1 the base moves slower than the
    # correction can follow. Can be changed on a resume: the optimiser state carries over per parameter.
    base_lr_scale: float = 1.0
    # Weak cost on the size of the correction near the surface (rounded_box base): weight x mean of
    # (f - base(x))^2 over the sample points with |f| < correction_cost_band (anchor rows left out).
    # f - base is the main network's own correction, so the cost pulls the correction toward zero (each
    # shape toward its base) and reaches the base only through the base features; the base is paid
    # indirectly: the other terms find it cheaper to move a shape through its base. 0 = off.
    # Bends and tapers of the rounded-box base (linear learned head only). 9 more head outputs, zero at
    # the start (no change): kappa_ij = base_bend_max tanh(u / base_bend_max) bends axis i toward axis j
    # (all six ordered pairs; the box is evaluated at y_j = p_j + kappa_ij p_i^2 / 2), and
    # t_i = base_taper_max tanh(u / base_taper_max) tapers along axis i (the other two coordinates
    # divided by 1 + t_i y_i). Tied to the fixed axes, never to "the longest axis", so nothing jumps
    # when extents cross. The deformed box is then recentred: its soft centroid (inside-weight
    # sigmoid(field / base_center_tau) over 2 x base_center_points fixed antithetic points in
    # [-0.9, 0.9]^3) is measured and the box shifted by it, differentiably. Finally the base is
    # intersected with the ball of radius base_max_reach, which keeps the outer guarantee.
    base_deform: bool = False
    base_bend_max: float = 2.0
    base_taper_max: float = 1.0
    base_center_points: int = 8192
    base_center_tau: float = 0.03
    # spread push targets (std across codes) for each bend and each taper parameter (0 = none)
    base_spread_target_bend: float = 0.0
    base_spread_target_taper: float = 0.0
    # Base repulsion: the main diversity term's mechanism (diversity_mode "nearest") applied to the
    # bases. Each code's base vector (extents and radius / base_repel_scale_shape, bends /
    # base_repel_scale_bend, tapers / base_repel_scale_taper) is pushed away from its nearest_k most
    # similar bases among the whole update's codes (the other passes as fixed neighbours, with their
    # own terms for the exact gradient), by -log(soft-min distance / base_repel_slope + repel_eps),
    # the soft minimum and distance (mean absolute difference) as in the main term. Fixed weight; 0 = off.
    base_repel_weight: float = 0.0
    base_repel_slope: float = 1.0
    base_repel_scale_shape: float = 0.1
    base_repel_scale_bend: float = 1.0
    base_repel_scale_taper: float = 0.3
    # Std of the start of the bend/taper rows of the base head's last layer (0 = zero). A repulsion
    # cannot separate bases that are all identical, so these rows need a tiny random start.
    base_deform_init_std: float = 0.0
    correction_cost_weight: float = 0.0
    correction_cost_band: float = 0.05
    # Gradient the slope (eikonal) and consistency terms send into the base parameters, as a factor of
    # the true one (the field and every other route are unchanged). Below 1 the base is less held to
    # shapes whose distance field is easy to make exact (the sphere); costs one extra partial backward
    # pass per sub-batch. 1 = off.
    base_eikonal_grad_scale: float = 1.0
    base_consistency_grad_scale: float = 1.0
    # Direct push for variety between bases: an adaptive constraint base_spread =
    # sum over the extents of relu(base_spread_target - std)^2 + relu(base_spread_target_r - std_r)^2,
    # std taken across the codes of each pass. Its weight starts at base_spread_mult_init and grows
    # while the spread is short of the targets. 0 = off.
    base_spread_target: float = 0.0
    base_spread_target_r: float = 0.0
    base_spread_mult_init: float = 1.0
    # Bounds of the spread push's adaptive weight: its ceiling, and the tolerance below which the
    # shortfall counts as met (so the weight relaxes). Measured in B14: with a 1e5 ceiling and a 1e-6
    # tolerance the weight hit the ceiling and stayed there, every clipped update became the spread
    # term and the run collapsed.
    base_spread_mult_max: float = 1e5
    base_spread_tolerance: float = 1e-6
    # The bend and taper spread targets grow linearly from 0 to their values between update
    # base_spread_ramp_from and base_spread_ramp_from + base_spread_ramp_steps (0 steps = no ramp).
    base_spread_ramp_from: int = 0
    base_spread_ramp_steps: int = 0
    # How the code sets the base. "learned": the head above. Measured in B1 (warm start from J4):
    # the head drifted below the soft ramp's narrow range (temperature 0.01) within about 100
    # updates and stayed there, the same for every code. "fixed": no weights. Four fixed random
    # directions of the code (an orthonormal projection drawn from base_map_seed) give
    # s = sqrt(latent_dim) P^T z, about N(0, 1) each for random codes, and
    #   r   = base_round_min + (base_map_round_max - base_round_min) sigmoid(round_gain s4 + round_bias)
    #   b_i = (reach - r) sigmoid(extent_gain s_i + extent_bias),  i = 1..3 (each axis on its own,
    #         so a rod can lie along any axis), scaled down together if |b| + r passes the reach.
    # The base then differs between codes by construction; the ramp still starts at the sphere.
    base_map: str = "learned"
    base_map_seed: int = 0
    # Defaults (20,000 random codes, 64 dims): rounding radius median 0.13 (the fitted anchors: 0.11),
    # 47% with no axis longer than 2r, 24% rods, 23% sheets, 6% boxes, 21% thinner than 0.12.
    base_map_extent_gain: float = 3.0
    base_map_extent_bias: float = 0.0
    base_map_round_gain: float = 1.2
    base_map_round_bias: float = -1.5
    base_map_round_max: Optional[float] = None   # None = base_radius
    # With base_map "fixed": fit a rounded box to every anchor object and set the four base
    # components of its code so that its base matches that fit (the rest of the code is kept,
    # rescaled to unit length). Without it an anchor gets whatever base its code happens to give.
    anchor_base_fit: bool = False
    # base_surface_points > 0 writes that many points near the base surfaces of a few of the
    # batch's codes over the last rows of the (shared) volume samples, with Gaussian offsets of
    # std base_surface_noise along the normal. Thin bases hold few uniform samples. These points
    # are not uniform, so terms that read the samples as a uniform draw of the box (pose moments,
    # volume) are biased by them; keep it small.
    base_surface_points: int = 0
    base_surface_noise: float = 0.02
    # Start a new run from the whole model of a checkpoint (a trainer checkpoint or a plain state
    # dict): every weight whose name and shape match is copied, input weights that gained columns
    # (base_features) keep the old columns and start the new ones at zero, and new parts (the base
    # head) keep their own start. The anchor codes and the multipliers are copied too when they
    # match. Step, optimiser and schedule start fresh.
    init_model_from: Optional[str] = None
    window_start: float = 0.7       # (use_window only) correction window is 1 up to here
    cull_far_field: bool = True     # (use_window only) skip the network where the window is zero

    # batch
    clusters: int = 8
    codes_per_cluster: int = 4
    points: int = 2048              # N, volume samples per code
    # Where the N samples go. "uniform": uniform in the box [-1, 1]^3 (the original). "shells":
    # random directions at radii in four bands out to bound_radius, 5 / 50 / 20 / 25 % of the
    # points in 0-0.2 / 0.2-0.55 / 0.55-0.7 / 0.7-R_b (uniform by volume inside each band), every
    # point counted equally. Nothing lies beyond R_b, where the field is fixed by construction.
    point_sampling: str = "uniform"
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
    # * A temporary cap. field_cap > 0 holds the multipliers named in field_cap_names at or below
    #   field_cap until update field_cap_until, then raises the cap geometrically to field_cap_release_to
    #   over field_cap_release_steps updates, after which the configured upper limit applies again.
    field_cap: float = 0.0
    field_cap_names: Tuple[str, ...] = ("eikonal",)
    field_cap_until: int = 0
    field_cap_release_steps: int = 0
    field_cap_release_to: float = 100.0
    # * Even use of the code. code_isotropy_weight > 0 adds losses.code_isotropy: a step of the same
    #   size in any code direction should change the scales and shifts handed to the main network
    #   by the same amount (variance of the log change over code_isotropy_directions random
    #   directions per code). It acts on the side network only and costs next to nothing. It does
    #   not by itself make the main network turn those changes into shape changes.
    code_isotropy_weight: float = 0.0
    code_isotropy_directions: int = 16
    # * A bias toward flat faces and sharp edges. crease_weight > 0 adds losses.crease: at
    #   crease_points surface points per code, how fast the surface normal turns over a step of
    #   crease_step along the surface (turn = 1 / radius on a ball), costed turn^2 / (turn^2 +
    #   crease_scale^2). Flat is free and the cost levels off at 1, so packing a turn into a sharp
    #   edge is cheaper than spreading it over a bend. crease_scale is the bend (1 / radius) that
    #   costs one half; bends above crease_scale / sqrt(3) are in the range where concentrating
    #   them pays, so 3 covers every radius under 0.58. Edges cannot get sharper than about
    #   crease_step, and crease_band (how close to the surface both points of a pair must be) has
    #   to be larger than crease_step or pairs across an edge are thrown away. Anchor codes are
    #   left out unless crease_on_anchors (the anchor objects fix their own shape). Costs second
    #   derivatives at 2 * crease_points extra points per code.
    crease_weight: float = 0.0
    crease_points: int = 256
    crease_step: float = 0.02
    crease_scale: float = 3.0
    crease_band: float = 0.04
    crease_on_anchors: bool = False
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
    # * variety_weight_max > 0 lets the weight move between variety_weight and that limit: each update it
    #   grows by variety_weight_up while the smoothed consistency term is below
    #   variety_consistency_ceiling and the smoothed topology terms (bridge, delete, fill, widen) are
    #   within their tolerances, and otherwise (or when the graph was off) shrinks by
    #   variety_weight_down. Accumulated updates only.
    variety_weight_max: float = 0.0
    variety_weight_up: float = 0.001
    variety_weight_down: float = 0.002
    variety_consistency_ceiling: float = 0.025
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
    # anchor_weight_ramp_steps > 0 brings the anchor pull in gradually: its weight rises linearly from
    # anchor_weight_start * anchor_weight to anchor_weight over that many updates. At update 0 the
    # anchor pull is 70 to 90% of the gradient, before the slope and consistency weights have risen.
    anchor_weight_ramp_steps: int = 0
    anchor_weight_start: float = 0.01
    # anchor_weight_hold_steps > 0 keeps the weight at its starting value for that many updates
    # before the ramp begins (with anchor_weight_start = 0 the pull is off until then). Measured
    # on the base-free decoder: the full pull shrinks every shape to 3% of its volume within 25
    # updates, because all codes still give the same field and are dragged to the average anchor.
    anchor_weight_hold_steps: int = 0
    # eikonal_all_points: the slope term counts every sample, also those within
    # consistency_min_depth of the surface that the fixed-surface consistency setting leaves out
    # of both field terms. Without the sphere base nothing else holds the slope at the surface.
    eikonal_all_points: bool = False
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
    # Step at which the cosine decay begins (the rate is held at lr until then). 0 = right after
    # the warm-up, the original schedule.
    lr_decay_start: int = 0
    # Learning rate of the main network's final layer (512 to 1) as a fraction of lr. Its weights are
    # about 0.003, so at the full rate Adam changes them by 1.5 to 3% of themselves per update,
    # 10 to 30 times more than any other layer, and that change is mostly the same for every code.
    # 1 = the same rate as the rest of the network.
    out_lr_scale: float = 1.0
    # weight_average_decay > 0 keeps a running average of the model's weights, updated after every
    # optimiser step (average <- decay * average + (1 - decay) * weights; 0.998 averages over about
    # 500 updates). It does not affect training. Every checkpoint save also writes
    # checkpoint_average_last.pt: the same resume state with the averaged weights as the model.
    weight_average_decay: float = 0.0
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
        if self.variety_weight_max > 0:
            if int(self.accumulate) <= 1:
                raise ValueError("variety_weight_max needs accumulate > 1")
            if not 0.0 < self.variety_weight <= self.variety_weight_max:
                raise ValueError("need 0 < variety_weight <= variety_weight_max")
            if not (0.0 < self.variety_weight_up < 1.0 and 0.0 < self.variety_weight_down < 1.0):
                raise ValueError("variety_weight_up and variety_weight_down must lie in (0, 1)")
        if self.point_sampling not in ("uniform", "shells"):
            raise ValueError("point_sampling must be 'uniform' or 'shells'")
        if self.point_sampling == "shells" and not self.bound_radius > 0.7:
            raise ValueError("point_sampling 'shells' needs bound_radius > 0.7 (its outer band starts there)")
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
        if self.base_shape not in ("sphere", "rounded_box"):
            raise ValueError("base_shape must be 'sphere' or 'rounded_box'")
        if self.base_shape == "rounded_box":
            if self.use_window or not self.sphere_base:
                raise ValueError("base_shape 'rounded_box' needs use_window=False and sphere_base=True")
            if not self.base_radius <= self.max_reach() <= self.bound_radius - self.correction_bound + 1e-9:
                # the base can reach base_max_reach, the correction A beyond it
                raise ValueError("need base_radius <= base_max_reach <= bound_radius - correction_bound")
            if not 0.0 < self.base_round_min < self.base_radius:
                raise ValueError("need 0 < base_round_min < base_radius")
            if self.base_inner_fraction is not None and not self.base_inner_fraction > 0.0:
                raise ValueError("need base_inner_fraction > 0")
            if not self.base_speed > 0.0:
                raise ValueError("need base_speed > 0")
            if self.base_head_net not in ("shared", "shared_mlp", "separate"):
                raise ValueError("base_head_net must be 'shared', 'shared_mlp' or 'separate'")
            if self.base_param_inputs and not self.base_features:
                raise ValueError("base_param_inputs needs base_features")
            if self.base_head_form not in ("squashed", "linear"):
                raise ValueError("base_head_form must be 'squashed' or 'linear'")
            if self.base_map not in ("learned", "fixed"):
                raise ValueError("base_map must be 'learned' or 'fixed'")
            top = self.base_radius if self.base_map_round_max is None else float(self.base_map_round_max)
            if self.base_map == "fixed" and not self.base_round_min < top <= self.max_reach():
                raise ValueError("need base_round_min < base_map_round_max <= base_max_reach")
            if self.base_map == "fixed" and self.latent_dim < 5:
                raise ValueError("base_map 'fixed' takes 4 code directions: need latent_dim >= 5")
        if self.base_budget > 0 and not (self.base_shape == "rounded_box" and self.base_map == "learned"
                                         and self.base_head_form == "linear"):
            raise ValueError("base_budget needs a rounded_box base with the 'linear' learned head")
        if self.base_deform and not (self.base_shape == "rounded_box" and self.base_map == "learned"
                                     and self.base_head_form == "linear"):
            raise ValueError("base_deform needs a rounded_box base with the 'linear' learned head")
        if self.base_deform and self.base_surface_points > 0:
            raise ValueError("base_surface_points does not follow base_deform; use one or the other")
        if self.anchor_base_fit and not (self.base_shape == "rounded_box" and self.base_map == "fixed"):
            raise ValueError("anchor_base_fit needs base_shape 'rounded_box' with base_map 'fixed'")

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

    def max_reach(self) -> float:
        """Largest reach |b| + r of a rounded-box base (``base_max_reach``, or the most the bound allows)."""
        if self.base_max_reach is not None:
            return float(self.base_max_reach)
        return float(self.bound_radius - self.correction_bound)

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
