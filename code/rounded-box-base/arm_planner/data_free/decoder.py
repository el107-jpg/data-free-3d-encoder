"""Modulated sphere decoder: ``f(x, z) = (R - rho) + A tanh(net(x, z) / A)``.

Positive inside. Every code starts close to the sphere of radius ``R``. The
correction is bounded by ``A < R``, so for every code and any weights
``f > 0`` where ``rho < R - A`` (non-empty, origin inside) and ``f < 0`` where
``rho > R + A`` (bounded). Any shape between those two balls has a true signed
distance within ``A`` of ``R - rho``, so a true distance field is representable
everywhere. There is no fixed far field: an exact ``R - rho`` beyond some radius
is not the distance field of a deformed shape, and the SDF losses would pull
every shape back to the sphere.

``cfg.base_shape="rounded_box"`` replaces the sphere by a rounded box set by the code:
``f(x, z) = base(x; b(z), r(z)) + A tanh(net(x, z) / A)``, with ``b`` the half-extents of the core box
and ``r`` the rounding radius. At ``b = 0, r = R`` it is the sphere above, and the head that sets them
starts at zero output, so a new decoder is identical to the sphere one. See ``rounded_box_field``.

``cfg.use_window=True`` selects the older variant
``(R - rho) + window(rho) * c`` whose correction fades to exactly zero at
``bound_radius``; it is kept for comparison only.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import DataFreeConfig


class Modulation(NamedTuple):
    """Per-code FiLM parameters (and base shape), computed once per code by ``prepare``."""

    scale: Tensor  # B×L×W, layer l computes (W h + b) * (1 + scale_l) + shift_l
    shift: Tensor  # B×L×W
    base: Optional[Tensor] = None  # B×4 (b1, b2, b3, r) of a rounded-box base; None for the sphere
    deform: Optional[Tensor] = None  # B×12 (6 bends, 3 tapers, 3 centring shift) with base_deform; else None

    def index(self, idx: Tensor) -> "Modulation":
        """Modulation of a subset (or a repetition) of the codes."""
        return Modulation(self.scale[idx], self.shift[idx], None if self.base is None else self.base[idx],
                          None if self.deform is None else self.deform[idx])


def rounded_box_field(x: Tensor, base: Tensor, eps: float = 1e-8) -> Tensor:
    """Signed distance of a rounded box, positive inside. ``x`` is B×M×3, ``base`` B×4 -> B×M.

    ``r - (|max(q, 0)| + min(max_i q_i, 0))`` with ``q = |x| - b``: the points within ``r`` of the
    core box ``|x_i| <= b_i``. Exact (slope 1) everywhere except on the core's medial set inside.
    At ``b = 0`` it is ``r - sqrt(|x|^2 + eps)``, the sphere field of the plain decoder to the bit.
    A slightly negative ``b_i`` (the head allows down to about -0.007) gives a lens, still slope 1.
    """
    b, r = base[:, None, :3], base[:, None, 3]
    q = x.abs() - b
    outside = torch.sqrt(q.clamp_min(0.0).square().sum(-1) + eps)
    inside = q.max(dim=-1).values.clamp_max(0.0)
    return r - (outside + inside)


def core_point(x: Tensor, base: Tensor) -> Tensor:
    """Nearest point of the core box ``|x_i| <= max(b_i, 0)`` to ``x`` (B×M×3), 1-Lipschitz in x."""
    b = base[:, None, :3].clamp_min(0.0)
    return torch.maximum(torch.minimum(x, b), -b)


# Temperature of the soft ramp that turns the head's output into half-extents: b(0) = 0 exactly,
# slope 1/2 there, about the identity for larger outputs, never below -BASE_TAU ln 2.
BASE_TAU = 0.01
BASE_FEATURES = 4  # base(x) and core_point(x)
BEND_PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))  # (i, j): axis i bends toward axis j


def deform_points(x: Tensor, deform: Tensor) -> Tensor:
    """Box coordinates of points ``x`` (B×M×3) under ``deform`` (B×12: 6 bends, 3 tapers, 3 shifts).

    Shift ``p = x + c``; bend ``y_j = p_j + (1/2) sum_i kappa_ij p_i^2``; taper: each coordinate ``k``
    divided by ``prod_{i != k} (1 + t_i y_i)`` (each factor kept at least 0.3, which only matters far
    outside the shape). All smooth in the parameters; all zero is the identity.
    """
    b = deform.shape[0]
    kappa = deform.new_zeros(b, 3, 3)
    for n, (i, j) in enumerate(BEND_PAIRS):
        kappa[:, i, j] = deform[:, n]
    p = x + deform[:, None, 9:12]
    y = p + 0.5 * torch.bmm(p.square(), kappa)
    s = (1.0 + deform[:, None, 6:9] * y).clamp_min(0.3)
    total = s.prod(dim=-1, keepdim=True)
    return y * s / total


def smooth_window(rho: Tensor, start: float, end: float) -> Tensor:
    """1 for ``rho <= start``, quintic smoothstep to exactly 0 for ``rho >= end`` (C2)."""
    u = ((end - rho) / (end - start)).clamp(0.0, 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


class ModulatedSphereDecoder(nn.Module):
    def __init__(self, cfg: DataFreeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.width = cfg.width
        self.num_layers = cfg.layers
        self.skip_layer = cfg.skip_layer if cfg.skip_layer and 0 < cfg.skip_layer < cfg.layers else None
        self.beta = float(cfg.softplus_beta)
        self.leak = float(getattr(cfg, "activation_leak", 0.0))
        normed = min(max(int(getattr(cfg, "layer_norm_layers", 0)), 0), cfg.layers)
        self.norm_from = cfg.layers - normed          # index of the first layer-normalised hidden layer
        self.base_radius = float(cfg.base_radius)
        self.window_start = float(cfg.window_start)
        self.bound_radius = float(cfg.bound_radius)
        self.correction_bound = float(cfg.correction_bound)
        inner = getattr(cfg, "correction_bound_inner", None)
        self.correction_inner = self.correction_bound if inner is None else float(inner)
        self.use_window = bool(cfg.use_window)
        self.sphere_base = bool(getattr(cfg, "sphere_base", True))
        self.outer_limit = str(getattr(cfg, "outer_limit", "blend"))
        if self.outer_limit not in ("blend", "min"):
            raise ValueError("outer_limit must be 'blend' or 'min'")
        self.cull_far_field = bool(cfg.cull_far_field)
        self.box_base = str(getattr(cfg, "base_shape", "sphere")) == "rounded_box"
        self.base_features = self.box_base and bool(getattr(cfg, "base_features", True))
        self.base_param_inputs = self.base_features and bool(getattr(cfg, "base_param_inputs", False))
        self.n_base_features = (BASE_FEATURES + (4 if self.base_param_inputs else 0)) if self.base_features else 0
        inner_fraction = getattr(cfg, "base_inner_fraction", None)
        self.base_inner_fraction = None if not self.box_base or inner_fraction is None else float(inner_fraction)

        # Mapping network: unit code -> FiLM scale and shift of every hidden layer.
        mapping = []
        d = cfg.latent_dim
        for _ in range(cfg.mapping_layers):
            mapping += [nn.Linear(d, cfg.mapping_width), nn.Softplus(beta=self.beta)]
            d = cfg.mapping_width
        self.mapping = nn.Sequential(*mapping)
        self.film = nn.Linear(d, 2 * cfg.layers * cfg.width)
        self.base_fixed = self.box_base and str(getattr(cfg, "base_map", "learned")) == "fixed"
        self.base_deform = self.box_base and bool(getattr(cfg, "base_deform", False))
        n_out = 13 if self.base_deform else 4
        if self.base_deform:
            self.bend_max = float(cfg.base_bend_max)
            self.taper_max = float(cfg.base_taper_max)
            self.center_tau = float(cfg.base_center_tau)
            # fixed antithetic quasi-random points: a symmetric shape's centroid comes out exactly 0
            sobol = torch.quasirandom.SobolEngine(3, scramble=True, seed=1234).draw(int(cfg.base_center_points))
            half = (sobol * 2.0 - 1.0) * 0.9
            self.register_buffer("center_points", torch.cat([half, -half]), persistent=False)
        if self.box_base:
            # Base head: mapping features -> (u1, u2, u3, u_r), zero at the start (the sphere).
            # base_range (a buffer, saved with the weights) is the ramp factor the trainer sets.
            # With base_map "fixed" there is no head: a fixed projection of the code (a buffer).
            if self.base_fixed:
                g = torch.Generator().manual_seed(int(getattr(cfg, "base_map_seed", 0)))
                q = torch.linalg.qr(torch.randn((cfg.latent_dim, 4), generator=g)).Q      # D x 4, orthonormal
                self.register_buffer("base_projection", q)
                self.map_extent = (float(cfg.base_map_extent_gain), float(cfg.base_map_extent_bias))
                self.map_round = (float(cfg.base_map_round_gain), float(cfg.base_map_round_bias))
                top = getattr(cfg, "base_map_round_max", None)
                self.map_round_max = float(cfg.base_radius if top is None else top)
            elif str(getattr(cfg, "base_head_net", "shared")) == "separate":
                width = int(getattr(cfg, "base_net_width", 64))
                self.base_net = nn.Sequential(nn.Linear(cfg.latent_dim, width), nn.Tanh(), nn.Linear(width, n_out))
            elif str(getattr(cfg, "base_head_net", "shared")) == "shared_mlp":
                width = int(getattr(cfg, "base_net_width", 64))
                self.base_head_mlp = nn.Sequential(nn.Linear(d, width), nn.Tanh(), nn.Linear(width, n_out))
            else:
                self.base_head = nn.Linear(d, n_out)
            self.base_linear = str(getattr(cfg, "base_head_form", "squashed")) == "linear"
            self.base_max_reach = float(cfg.max_reach())
            self.base_round_min = float(cfg.base_round_min)
            self.base_speed = float(getattr(cfg, "base_speed", 1.0))
            ramp = int(getattr(cfg, "base_ramp_steps", 0))
            self.register_buffer("base_range", torch.tensor(0.0 if ramp > 0 else 1.0))

        # Trunk on the coordinates (raw, or with fourier_bands sines and cosines of them added), with
        # the same input concatenated back in at skip_layer.
        self.fourier_bands = max(int(getattr(cfg, "fourier_bands", 0)), 0)
        self.register_buffer("fourier_freq", math.pi * 2.0 ** torch.arange(self.fourier_bands, dtype=torch.float32),
                             persistent=False)
        self.input_dim = 3 + 6 * self.fourier_bands + self.n_base_features
        self.hidden = nn.ModuleList()
        for layer in range(cfg.layers):
            fan_in = self.input_dim if layer == 0 else cfg.width
            if layer == self.skip_layer:
                fan_in += self.input_dim
            self.hidden.append(nn.Linear(fan_in, cfg.width))
        self.out = nn.Linear(cfg.width, 1)
        self.reset_parameters()
        self.weight_norm = bool(getattr(cfg, "weight_norm", False))
        if self.weight_norm:
            # Every unit of the main network stores its incoming weights as a direction v and a size
            # g (w = g v / |v|). The function at the start is unchanged. All directions are given the
            # same length (the hidden layers' sqrt(2)), so one optimiser step turns a unit of the
            # output layer, whose weights are tiny, by the same angle as a unit of a hidden layer.
            for lin in list(self.hidden) + [self.out]:
                nn.utils.parametrizations.weight_norm(lin, name="weight", dim=0)
                with torch.no_grad():
                    v = lin.parametrizations.weight.original1
                    v.mul_(math.sqrt(2.0) / v.norm(dim=1, keepdim=True).clamp_min(1e-12))

    def reset_parameters(self) -> None:
        for lin in self.hidden:
            # Softplus(beta=20) is close to ReLU, so use the ReLU gain. Biases
            # keep the default uniform init, which spreads the bends over the box.
            nn.init.normal_(lin.weight, std=math.sqrt(2.0 / lin.in_features))
        # Output small but not tiny: every code starts close to the sphere, and
        # different codes start measurably different (the diversity loss is a
        # mean absolute difference between fields, with zero gradient when two
        # fields are identical).
        nn.init.normal_(self.out.weight, std=self.cfg.output_init_std)
        nn.init.zeros_(self.out.bias)
        if self.box_base and not self.base_fixed:
            if hasattr(self, "base_net"):
                last = self.base_net[-1]
            elif hasattr(self, "base_head_mlp"):
                last = self.base_head_mlp[-1]
            else:
                last = self.base_head
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            std = float(getattr(self.cfg, "base_deform_init_std", 0.0))
            if self.base_deform and std > 0:
                # a tiny random start of the bend/taper rows breaks the symmetry a repulsion needs
                with torch.no_grad():
                    gen = torch.Generator().manual_seed(4242)
                    last.weight[4:].copy_(torch.randn(last.weight[4:].shape, generator=gen) * std)
        if self.box_base:
            if self.base_features:
                # The base features enter at zero weight (the last BASE_FEATURES input columns of
                # the first layer and of the skip layer), so they change nothing at the start.
                with torch.no_grad():
                    self.hidden[0].weight[:, -self.n_base_features:].zero_()
                    if self.skip_layer is not None:
                        self.hidden[self.skip_layer].weight[:, -self.n_base_features:].zero_()
        # FiLM head: calibrated on random probe codes so that every scale and
        # shift has mean 0 and standard deviation film_init_std across codes.
        nn.init.normal_(self.film.weight, std=1.0)
        nn.init.zeros_(self.film.bias)
        std = float(self.cfg.film_init_std)
        with torch.no_grad():
            if std <= 0.0:
                self.film.weight.zero_()
                return
            ref = self.film.weight
            probe = F.normalize(torch.randn(1024, self.latent_dim, device=ref.device, dtype=ref.dtype), dim=-1)
            raw = self.film(self.mapping(probe))
            gain = std / raw.std(dim=0).clamp_min(1e-12)
            self.film.weight.mul_(gain[:, None])
            self.film.bias.copy_(-raw.mean(dim=0) * gain)

    # ------------------------------------------------------------------ codes
    def prepare(self, codes: Tensor) -> Modulation:
        """Normalise ``codes`` (B×D) to unit length and compute their modulation."""
        z = F.normalize(codes, dim=-1)
        features = self.mapping(z)
        film = self.film(features)
        scale, shift = film.view(z.shape[0], 2, self.num_layers, self.width).unbind(1)
        if not self.box_base:
            base = None
        elif self.base_fixed:
            base = self.fixed_base(z)
        else:
            base = self.base_parameters(features, z)
        deform = None
        if self.base_deform:
            base, deform = base[:, :4], self._deform_parameters(base, base[:, 4:])
        return Modulation(scale, shift, base, deform)

    def _deform_parameters(self, base: Tensor, u: Tensor) -> Tensor:
        """Bends, tapers (bounded by tanh, slope 1 at 0, scaled by the ramp) and the centring shift.

        The shift is the soft centroid of the deformed, unshifted box over the fixed antithetic points,
        so the shifted box has centroid 0 (to the accuracy of those points). Differentiable throughout.
        """
        t = self.base_range.to(u.dtype)
        bend = t * self.bend_max * torch.tanh(u[:, :6] / self.bend_max)
        taper = t * self.taper_max * torch.tanh(u[:, 6:9] / self.taper_max)
        zero = torch.zeros_like(taper)
        raw = torch.cat([bend, taper, zero], dim=-1)
        pts = self.center_points.to(u.dtype)[None].expand(u.shape[0], -1, -1)
        inside = torch.sigmoid(rounded_box_field(deform_points(pts, raw), base) / self.center_tau)
        centre = (inside[..., None] * pts).sum(1) / inside.sum(1).clamp_min(1e-6)[:, None]
        return torch.cat([bend, taper, centre], dim=-1)

    def base_scores(self, z: Tensor) -> Tensor:
        """The four base components of unit codes ``z`` (B×D): ``s = sqrt(D) P^T z``, about N(0, 1) each."""
        return math.sqrt(self.latent_dim) * (z @ self.base_projection.to(z.dtype))

    def fixed_base(self, z: Tensor) -> Tensor:
        """Rounded-box base (b1, b2, b3, r), B×4, as a fixed function of unit codes (``base_map="fixed"``).

        See ``DataFreeConfig.base_map``. Ramped from the sphere by ``base_range`` like the learned head.
        """
        s = self.base_scores(z)
        low, top, reach = self.base_round_min, self.map_round_max, self.base_max_reach
        r = low + (top - low) * torch.sigmoid(self.map_round[0] * s[:, 3] + self.map_round[1])
        room = reach - r
        b = room[:, None] * torch.sigmoid(self.map_extent[0] * s[:, :3] + self.map_extent[1])
        size = torch.sqrt(b.square().sum(-1) + 1e-12)
        b = b * torch.clamp(room / size, max=1.0)[:, None]
        t = self.base_range.to(b.dtype)
        sphere = self.base_radius
        return torch.cat([t * b, (sphere + t * (r - sphere))[:, None]], dim=-1)

    def scores_for_base(self, base: Tensor, limit: float = 2.5, extent_max: float = 0.95,
                        round_margin: float = 0.02) -> Tensor:
        """Base components ``s`` (K×4) whose fixed map gives ``base`` (K×4, at full ramp), within limits.

        Exact when the target is inside the map's range and its corner is within the reach.
        Targets beyond it are met only part of the way: an extent at most ``extent_max`` of the
        room it has, the rounding radius inside the middle ``1 - 2 round_margin`` of its range,
        and every component within ±``limit``. Without that, objects larger or thinner than the
        map allows ask for components of 3 or more, which take most of a unit code's length (a
        random code puts about 4 / D of it there) and push the code far from where it was.
        """
        low, top, reach = self.base_round_min, self.map_round_max, self.base_max_reach
        frac = ((base[:, 3] - low) / (top - low)).clamp(round_margin, 1 - round_margin)
        s_r = (torch.logit(frac) - self.map_round[1]) / self.map_round[0]
        r = low + (top - low) * frac
        a = (base[:, :3].clamp_min(0.0) / (reach - r)[:, None]).clamp(1e-3, extent_max)
        s_b = (torch.logit(a) - self.map_extent[1]) / self.map_extent[0]
        return torch.cat([s_b, s_r[:, None]], dim=-1).clamp(-limit, limit)

    def base_parameters(self, features: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Rounded-box base (b1, b2, b3, r), B×4, from the mapping network's output.

        The head gives ``u`` (zero at the start). The radius is ``r_min + (reach - r_min) σ(u_r + c)``
        with ``c`` set so that ``u_r = 0`` gives ``R``. The extents are cumulative sums of soft ramps
        ``p_i = τ (softplus(u_i / τ) - ln 2)`` (``p(0) = 0``): ``b3 = p3, b2 = b3 + p2, b1 = b2 + p1``,
        ordered up to ``τ ln 2``, so a plate or rod always lies along the same axes and swapping axes
        cannot pass for variety. The farthest point of the base from the origin is a corner of the
        core grown by r, at ``|b| + r``; if that would pass the reach limit, all three extents are
        scaled down together (so a rod may reach the limit along its axis, a cube only along its
        diagonal). Finally the deviation from the sphere is scaled by ``base_range``:
        ``b <- t b``, ``r <- R + t (r - R)``, which keeps ``|b| + r <= R + t (reach - R)``.
        """
        reach, low, sphere = self.base_max_reach, self.base_round_min, self.base_radius
        if hasattr(self, "base_net"):
            # separate base network on the unit code (``base_head_net="separate"``)
            u = self.base_speed * self.base_net(z)
        elif hasattr(self, "base_head_mlp"):
            # the side network's features through one more nonlinear layer (``base_head_net="shared_mlp"``)
            u = self.base_speed * self.base_head_mlp(features)
        else:
            u = self.base_speed * self.base_head(features)
        if self.base_linear:
            return self._linear_base(u)
        u = u[:, :4]
        start = (sphere - low) / (reach - low)
        shift = math.log(start / (1.0 - start)) if start < 1.0 else 20.0
        r = low + (reach - low) * torch.sigmoid(u[:, 3] + shift)
        p = BASE_TAU * (F.softplus(u[:, :3] / BASE_TAU) - math.log(2.0))
        b = p.flip(-1).cumsum(-1).flip(-1)                       # (p1+p2+p3, p2+p3, p3)
        room = reach - r
        size = torch.sqrt(b.clamp_min(0.0).square().sum(-1) + 1e-12)
        over = size > room
        cap = torch.where(over, room / torch.where(over, size, torch.ones_like(room)), torch.ones_like(room))
        b = b * cap[:, None]
        t = self.base_range.to(b.dtype)
        return torch.cat([t * b, (sphere + t * (r - sphere))[:, None]], dim=-1)

    def _linear_base(self, u: Tensor) -> Tensor:
        """``base_head_form="linear"``: the head output is the deviation from the sphere.

        ``b_i = u_i`` and ``r = R + u_r`` (slope 1, exactly the sphere at ``u = 0``) except near soft
        limits that only keep the base meaningful: extents stay above about -0.1 (a lens; the loss
        penalises any negative extent long before that), r above ``base_round_min`` and below the
        reach (+7e-5). Temperatures 0.02, 0.05 and 0.01; the slope at the start is 1, 1 and 0.99.
        At r = R the corner cap leaves the extents only reach - R (0.05 at the defaults) of room, so
        a base stretches by growing extents and shrinking r together. The
        corner ``|b| + r`` is capped at the reach (all extents scaled down together; the hard limit
        that keeps the outer guarantee). The ramp ``base_range`` scales the deviation as before.
        """
        sphere, low, reach = self.base_radius, self.base_round_min, self.base_max_reach
        extra = u[:, 4:]
        u = u[:, :4]
        floor, tb = -0.1, 0.02
        b = floor + tb * F.softplus((u[:, :3] - floor) / tb)
        b = b - (floor + tb * F.softplus(torch.tensor(-floor / tb))).to(b)            # exactly 0 at u = 0
        tr, tc = 0.05, 0.01
        r = low + tr * F.softplus((sphere + u[:, 3] - low) / tr)
        r = r - (low + tr * F.softplus(torch.tensor((sphere - low) / tr)) - sphere).to(r)   # exactly R at u = 0
        # soft ceiling at the reach (slope 0.99 at the sphere), shifted so u = 0 is still exactly R
        r = reach - tc * F.softplus((reach - r) / tc)
        r = r + (tc * F.softplus(torch.tensor((reach - sphere) / tc)) - (reach - sphere)).to(r)
        room = (reach - r).clamp_min(0.0)
        size = torch.sqrt(b.clamp_min(0.0).square().sum(-1) + 1e-12)
        over = size > room
        cap = torch.where(over, room / torch.where(over, size, torch.ones_like(room)), torch.ones_like(room))
        b = b * cap[:, None]
        t = self.base_range.to(b.dtype)
        return torch.cat([t * b, (sphere + t * (r - sphere))[:, None], extra], dim=-1)

    def base_field(self, mod: Modulation, x: Tensor) -> Tensor:
        """The base field at ``x`` (B×M×3) -> B×M: the sphere, or the code's (deformed) rounded box.

        With deformations the box is evaluated at the deformed coordinates and intersected with the
        ball of radius base_max_reach (so the base never reaches past it, whatever the bends).
        """
        if mod.base is None:
            return self.base_radius - torch.sqrt((x * x).sum(-1) + 1e-8)
        if mod.deform is None:
            return rounded_box_field(x, mod.base)
        box = rounded_box_field(deform_points(x, mod.deform), mod.base)
        ball = self.base_max_reach - torch.sqrt((x * x).sum(-1) + 1e-8)
        return torch.minimum(box, ball)

    def base_core(self, mod: Modulation, x: Tensor) -> Tensor:
        """Nearest point of the core box, in the box's own (deformed) coordinates (a main-network input)."""
        y = x if mod.deform is None else deform_points(x, mod.deform)
        return core_point(y, mod.base)

    # ------------------------------------------------------------------ field
    def encode(self, x: Tensor) -> Tensor:
        """The main network's input: x itself, plus sin and cos of pi 2^k x for k < fourier_bands.

        Band k has a period of 2 / 2^k in a box of side 2, so 3 bands reach a period of 0.5 and 5 bands
        0.125. More bands let the network draw finer detail and also finer ripple.
        """
        if self.fourier_bands == 0:
            return x
        angle = x[..., None] * self.fourier_freq                       # ... x 3 x bands
        return torch.cat([x, torch.sin(angle).flatten(-2), torch.cos(angle).flatten(-2)], dim=-1)

    def activation(self, value: Tensor) -> Tensor:
        """Softplus, or with ``activation_leak`` a softplus that keeps slope ``leak`` when off."""
        out = F.softplus(value, beta=self.beta)
        if self.leak > 0.0:
            # out + leak * (value - out), without temporaries: in place when no gradient is being
            # recorded (the large evaluation passes), one extra tensor otherwise.
            if out.requires_grad:
                out = torch.lerp(out, value, self.leak)
            else:
                out.lerp_(value, self.leak)
        return out

    def _net(self, mod: Modulation, x: Tensor, collect: Optional[list] = None) -> Tensor:
        """Raw correction network. ``x`` is B×M×3, the result B×M.

        ``collect`` (a list) receives every hidden layer's input to the activation, B×M×W each,
        for measurements of which units are in use.
        """
        # (W h + b) * (1 + scale) + shift is applied as one batched affine map
        # with per-code weights W * (1 + scale) and bias b * (1 + scale) + shift.
        # The weights are only B×W×W, and autograd then keeps one B×M×W tensor
        # per layer less than the unfolded form (measured: about 25% less peak
        # memory and time for the double-backward step).
        gain = 1.0 + mod.scale  # B×L×W
        if self.base_features:
            # The extra inputs come last, so the columns of a decoder without them line up.
            parts = [self.base_field(mod, x)[..., None], self.base_core(mod, x)]
            if self.base_param_inputs:
                # the base parameters themselves, the same at every point of a shape
                own = torch.cat([mod.base[:, :3], mod.base[:, 3:] - self.base_radius], dim=-1)
                parts.append(own[:, None, :].expand(-1, x.shape[1], -1))
            extra = torch.cat(parts, dim=-1)
            x = torch.cat([self.encode(x), extra], dim=-1)
        else:
            x = self.encode(x)
        h = x
        for layer, lin in enumerate(self.hidden):
            if layer == self.skip_layer:
                h = torch.cat([h, x], dim=-1)
            g = gain[:, layer]
            if layer >= self.norm_from:
                # Layer norm: at every point the W values of W h + b are shifted and scaled to mean
                # 0 and variance 1 (across the units, not across points, so the field stays a plain
                # function of code and position); the scale and shift of the code are applied after.
                plain = F.layer_norm(F.linear(h, lin.weight, lin.bias), (self.width,))
                pre = torch.addcmul(mod.shift[:, layer][:, None, :], plain, g[:, None, :])   # one tensor, not two
            else:
                weight_t = lin.weight.t()[None] * g[:, None, :]          # B×in×W
                bias = (lin.bias * g + mod.shift[:, layer])[:, None, :]  # B×1×W
                pre = torch.baddbmm(bias, h, weight_t)
            if collect is not None:
                collect.append(pre)
            h = self.activation(pre)
        return self.out(h).squeeze(-1)

    def correction(self, mod: Modulation, x: Tensor) -> Tensor:
        """Bounded correction ``c = A tanh(net / A)`` (before any window), B×M.

        With ``correction_bound_inner`` the bound is ``A`` for positive values (towards inside,
        which keeps the shape within radius ``R + A``) and the inner bound for negative ones.
        Both branches have slope 1 and zero curvature at 0, so the join is smooth.
        """
        a, b = self.correction_bound, self.correction_inner
        raw = self._net(mod, x)
        if self.base_inner_fraction is not None and mod.base is not None:
            # Per code: the inward bound is k r, so the core box grown by (1 - k) r stays inside.
            b = (self.base_inner_fraction * mod.base[:, 3])[:, None]
            return torch.where(raw >= 0, a * torch.tanh(raw / a), b * torch.tanh(raw / b))
        if b == a:
            return a * torch.tanh(raw / a)
        return torch.where(raw >= 0, a * torch.tanh(raw / a), b * torch.tanh(raw / b))

    def query(self, mod: Modulation, x: Tensor) -> Tensor:
        """Field at ``x`` (B×N×3, or B×...×3) for prepared codes. Returns B×N (B×...)."""
        lead = x.shape[:-1]
        x = x.reshape(x.shape[0], -1, 3)
        if self.box_base:
            return (self.base_field(mod, x) + self.correction(mod, x)).reshape(lead)
        rho = torch.sqrt((x * x).sum(-1) + 1e-8)
        base = self.base_radius - rho
        plain = not self.sphere_base
        if not self.use_window and not plain:
            return (base + self.correction(mod, x)).reshape(lead)
        if plain and self.outer_limit == "min":
            # The network's output everywhere, cut off by the bounding ball: a smooth minimum of the
            # two fields (the intersection of the shape with the ball). Nothing is pinned inside.
            k = 0.02
            pair = torch.stack([self._net(mod, x), self.bound_radius - rho], dim=-1)
            return (-k * torch.logsumexp(-pair / k, dim=-1)).reshape(lead)

        window = smooth_window(rho, self.window_start, self.bound_radius)

        m = x.shape[1]
        keep = None
        if self.cull_far_field and m > 0:
            # The window is exactly zero from the bounding radius on, so the
            # network is only evaluated on the points inside it. They are packed
            # to the largest count in the batch; padding slots hold far points,
            # whose window is zero.
            near = rho < self.bound_radius
            k = int(near.sum(dim=1).max())
            if k == 0:
                return base.reshape(lead)
            if k < m:
                keep = torch.topk(near.to(torch.uint8), k, dim=1).indices

        # With the sphere as base the window fades the bounded correction out. Without it
        # (sphere_base off) the network's own output is the field where the window is 1 and is
        # blended into the sphere's field towards the bounding radius: f = w net + (1 - w)(R - rho).
        if keep is None:
            inner = self._net(mod, x) - base if plain else self.correction(mod, x)
            out = base + window * inner
        else:
            xs = torch.gather(x, 1, keep[..., None].expand(-1, -1, 3))
            inner = self._net(mod, xs) - torch.gather(base, 1, keep) if plain else self.correction(mod, xs)
            cs = torch.gather(window, 1, keep) * inner
            out = base + torch.zeros_like(base).scatter(1, keep, cs)
        return out.reshape(lead)

    def forward(self, codes: Tensor, x: Tensor) -> Tensor:
        return self.query(self.prepare(codes), x)
