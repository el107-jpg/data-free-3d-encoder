"""Modulated sphere decoder: ``f(x, z) = (R - rho) + A tanh(net(x, z) / A)``.

Positive inside. Every code starts close to the sphere of radius ``R``. The
correction is bounded by ``A < R``, so for every code and any weights
``f > 0`` where ``rho < R - A`` (non-empty, origin inside) and ``f < 0`` where
``rho > R + A`` (bounded). Any shape between those two balls has a true signed
distance within ``A`` of ``R - rho``, so a true distance field is representable
everywhere. There is no fixed far field: an exact ``R - rho`` beyond some radius
is not the distance field of a deformed shape, and the SDF losses would pull
every shape back to the sphere.

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
    """Per-code FiLM parameters, computed once per code by ``prepare``."""

    scale: Tensor  # B×L×W, layer l computes (W h + b) * (1 + scale_l) + shift_l
    shift: Tensor  # B×L×W

    def index(self, idx: Tensor) -> "Modulation":
        """Modulation of a subset (or a repetition) of the codes."""
        return Modulation(self.scale[idx], self.shift[idx])


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
        self.cull_far_field = bool(cfg.cull_far_field)

        # Mapping network: unit code -> FiLM scale and shift of every hidden layer.
        mapping = []
        d = cfg.latent_dim
        for _ in range(cfg.mapping_layers):
            mapping += [nn.Linear(d, cfg.mapping_width), nn.Softplus(beta=self.beta)]
            d = cfg.mapping_width
        self.mapping = nn.Sequential(*mapping)
        self.film = nn.Linear(d, 2 * cfg.layers * cfg.width)

        # Trunk on the coordinates (raw, or with fourier_bands sines and cosines of them added), with
        # the same input concatenated back in at skip_layer.
        self.fourier_bands = max(int(getattr(cfg, "fourier_bands", 0)), 0)
        self.register_buffer("fourier_freq", math.pi * 2.0 ** torch.arange(self.fourier_bands, dtype=torch.float32),
                             persistent=False)
        self.input_dim = 3 + 6 * self.fourier_bands
        self.hidden = nn.ModuleList()
        for layer in range(cfg.layers):
            fan_in = self.input_dim if layer == 0 else cfg.width
            if layer == self.skip_layer:
                fan_in += self.input_dim
            self.hidden.append(nn.Linear(fan_in, cfg.width))
        self.out = nn.Linear(cfg.width, 1)
        self.reset_parameters()

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
        film = self.film(self.mapping(z))
        scale, shift = film.view(z.shape[0], 2, self.num_layers, self.width).unbind(1)
        return Modulation(scale, shift)

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
        if b == a:
            return a * torch.tanh(raw / a)
        return torch.where(raw >= 0, a * torch.tanh(raw / a), b * torch.tanh(raw / b))

    def query(self, mod: Modulation, x: Tensor) -> Tensor:
        """Field at ``x`` (B×N×3, or B×...×3) for prepared codes. Returns B×N (B×...)."""
        lead = x.shape[:-1]
        x = x.reshape(x.shape[0], -1, 3)
        rho = torch.sqrt((x * x).sum(-1) + 1e-8)
        base = self.base_radius - rho
        if not self.use_window:
            return (base + self.correction(mod, x)).reshape(lead)

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

        if keep is None:
            out = base + window * self.correction(mod, x)
        else:
            xs = torch.gather(x, 1, keep[..., None].expand(-1, -1, 3))
            cs = torch.gather(window, 1, keep) * self.correction(mod, xs)
            out = base + torch.zeros_like(base).scatter(1, keep, cs)
        return out.reshape(lead)

    def forward(self, codes: Tensor, x: Tensor) -> Tensor:
        return self.query(self.prepare(codes), x)
