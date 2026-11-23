"""Trainer of the data-free latent space.

One *batch* is a set of latent clusters plus the volume samples each cluster
shares. Every batch gets ``cfg.graph_reuse_steps`` optimiser steps on the same
codes and points; the point graph is built on the first of them and reused.

One optimiser step (:meth:`Trainer.step`):

1. ``evaluate_field`` (two network passes, spatial gradients by autograd);
2. the SDF, validity, pose and diversity terms from that one evaluation;
3. the consistency error ``e`` picks the graph mode (``off`` / ``sign`` /
   ``full``); on the first step of a batch the graph is built and confirmed,
   and its hard negatives go into the anchor bank;
4. the five action losses on the graph's findings;
5. diversity is the objective, every other term a constraint weighted by an
   adaptive multiplier (``grad_change`` has a small fixed weight instead);
6. AdamW step with gradient-norm clipping, warm-up and cosine decay.

No mixed precision: autocast corrupts the spatial gradients the losses use.

Public API: :class:`AnchorBank`, :class:`Batch`, :class:`Trainer` (``step``,
``run``, ``run_diagnostics``, ``save_checkpoint``, ``from_checkpoint``),
:func:`train_step`, :func:`graph_mode_for`, :func:`lr_factor`,
:func:`parse_override`, :func:`apply_overrides`, :func:`main`.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import time
import typing
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from . import losses as L
from .config import DataFreeConfig
from .decoder import ModulatedSphereDecoder
from .field import FieldBatch, evaluate_field, field_and_grad, sample_points
from .graph import ShapeGraph, build_graph
from .graph_losses import ACTION_KEYS, action_losses, confirm, hard_negatives
from .latent import sample_cluster, sample_sphere, tangent_basis
from .multipliers import AdaptiveMultipliers

__all__ = [
    "CONSTRAINT_KEYS",
    "AnchorBank",
    "Batch",
    "Trainer",
    "train_step",
    "graph_mode_for",
    "lr_factor",
    "parse_override",
    "apply_overrides",
    "main",
]

# Constraints with an adaptive multiplier (names as in ``cfg.tolerances``).
FIELD_KEYS = ("eikonal", "consistency", "barrier", "origin", "volume") + L.POSE_KEYS
CONSTRAINT_KEYS = FIELD_KEYS + ACTION_KEYS
GRAPH_MODES = ("off", "sign", "full")
CHECKPOINT_NAME = "checkpoint_last.pt"


# --------------------------------------------------------------------- overrides
def _convert(text: str, hint: Any) -> Any:
    """``text`` converted to the type of a config field annotated ``hint``."""
    origin = typing.get_origin(hint)
    if origin is typing.Union:  # Optional[...]
        if text.strip().lower() in ("none", "null", ""):
            return None
        inner = [a for a in typing.get_args(hint) if a is not type(None)]
        return _convert(text, inner[0])
    if origin in (dict, Dict):
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object, got {text!r}")
        return value
    if hint is bool:
        low = text.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {text!r}")
    if hint is int:
        number = float(text)
        if number != int(number):
            raise ValueError(f"not an integer: {text!r}")
        return int(number)
    if hint is float:
        return float(text)
    return text


def parse_override(item: str) -> typing.Tuple[str, Any]:
    """``"key=value"`` -> ``(key, value)`` with the value typed like the config field.

    ``key`` is a :class:`DataFreeConfig` field, or one entry of a table:
    ``tolerances.<name>=x``, ``mult_rates.<name>=x``, ``mult_ranges.<name>=low,high``.
    """
    if "=" not in item:
        raise ValueError(f"override must look like key=value, got {item!r}")
    key, text = item.split("=", 1)
    key = key.strip()
    hints = typing.get_type_hints(DataFreeConfig)
    if "." in key:
        head, _ = key.split(".", 1)
        if head in ("tolerances", "mult_rates"):
            return key, float(text)
        if head == "mult_ranges":  # mult_ranges.delete=1,1e4
            low, high = (float(part) for part in text.split(","))
            return key, (low, high)
        raise KeyError(f"unknown config field {key!r}")
    if key not in hints:
        raise KeyError(f"unknown config field {key!r}")
    return key, _convert(text, hints[key])


def apply_overrides(cfg: DataFreeConfig, items: Optional[Sequence[str]]) -> DataFreeConfig:
    """A new config with ``key=value`` overrides applied (and re-validated)."""
    data = cfg.to_dict()
    for item in items or ():
        key, value = parse_override(item)
        if "." in key:
            head, name = key.split(".", 1)
            data[head] = dict(data[head])
            data[head][name] = value
        else:
            data[key] = value
    return DataFreeConfig.from_dict(data)


# ---------------------------------------------------------------------- schedule
def lr_factor(step: int, cfg: DataFreeConfig) -> float:
    """Learning-rate factor at optimiser step ``step``: linear warm-up, cosine to a floor."""
    warm = max(int(cfg.lr_warmup_steps), 0)
    if step < warm:
        return (step + 1) / warm
    start = max(warm, int(getattr(cfg, "lr_decay_start", 0) or 0))
    if step < start:
        return 1.0
    horizon = getattr(cfg, "lr_decay_steps", None)
    span = max(int(horizon if horizon else cfg.steps) - start, 1)
    progress = min((step - start) / span, 1.0)
    floor = float(cfg.lr_floor_fraction)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


class _Schedule:
    """Picklable ``lr_lambda`` for ``LambdaLR``."""

    def __init__(self, cfg: DataFreeConfig) -> None:
        self.cfg = cfg

    def __call__(self, step: int) -> float:
        return lr_factor(step, self.cfg)


def graph_mode_for(e: float, cfg: DataFreeConfig) -> str:
    """``off`` above ``graph_sign_error``, ``sign`` above ``graph_full_error``, else ``full``."""
    if not math.isfinite(e) or e > float(cfg.graph_sign_error):
        return "off"
    if e > float(cfg.graph_full_error):
        return "sign"
    return "full"


# ------------------------------------------------------------------- anchor bank
class AnchorBank:
    """Fixed unit codes, each with a NaN-padded cache of hard-negative points."""

    def __init__(self, cfg: DataFreeConfig, generator: torch.Generator, device) -> None:
        self.codes = sample_sphere(int(cfg.bank_size), cfg.latent_dim, generator, device)
        self.negatives = torch.full(
            (int(cfg.bank_size), int(cfg.hard_negatives), 3), float("nan"), device=device
        )

    @property
    def size(self) -> int:
        return self.codes.shape[0]

    def read(self, index: Tensor) -> Tensor:
        """Cached points of anchors ``index`` (``K×H×3``, NaN-padded)."""
        return self.negatives[index]

    @torch.no_grad()
    def write(self, index: Tensor, points: Tensor) -> None:
        """Merge new hard negatives (``K×H×3``, NaN-padded) into the cache.

        New points come first; cached points that are not repeated by a new
        one fill the remaining slots, so a feature found once stays visible
        until newer findings displace it.
        """
        capacity = self.negatives.shape[1]
        if index.numel() == 0 or capacity == 0:
            return
        new = points.to(self.negatives.dtype)
        old = self.negatives[index]
        new_ok = torch.isfinite(new).all(dim=-1)
        old_ok = torch.isfinite(old).all(dim=-1)
        # A cached point was fed into this batch, so it can come back as "new".
        gap = torch.cdist(torch.nan_to_num(old, nan=9.0), torch.nan_to_num(new, nan=-9.0))
        old_ok &= ~((gap < 1e-6) & new_ok[:, None, :]).any(dim=-1)
        merged = torch.cat([new, old], dim=1)
        ok = torch.cat([new_ok, old_ok], dim=1)
        order = torch.argsort((~ok).to(torch.int8), dim=1, stable=True)[:, :capacity]
        kept = torch.gather(merged, 1, order[..., None].expand(-1, -1, 3))
        kept_ok = torch.gather(ok, 1, order)
        self.negatives[index] = torch.where(
            kept_ok[..., None], kept, torch.full_like(kept, float("nan"))
        )

    def state_dict(self) -> dict:
        return {"codes": self.codes.cpu(), "negatives": self.negatives.cpu()}

    def load_state_dict(self, state: dict) -> None:
        self.codes = state["codes"].to(self.codes.device)
        self.negatives = state["negatives"].to(self.codes.device)


@dataclass
class Batch:
    """Sampled clusters and their shared points, plus the graph built on them."""

    codes: Tensor               # B×D, cluster by cluster, centre first
    cluster: Tensor             # B long
    x: Tensor                   # C×N×3 shared volume samples
    bank_index: Tensor          # C long, anchor index of the centre, -1 when fresh
    eps: Tensor                 # C chord radius of each cluster
    inner: int = 0              # optimiser steps already taken on this batch
    mode: str = "off"           # graph mode chosen on the first step
    graph: Optional[ShapeGraph] = None
    pair_first: Optional[Tensor] = None    # rows of codes that have a weak-direction partner
    pair_second: Optional[Tensor] = None   # rows of their partners (the last rows of codes)
    pair_chord: Optional[Tensor] = None    # chord between the two codes of each pair
    frame_used: int = -1                   # >= 0: the partners form a frame around code 0, this many along used directions
    anchor_index: Optional[Tensor] = None  # anchor objects whose codes are the last rows of codes
    anchor_pull: Optional[Tensor] = None   # weight of each of those rows (None = 1), smaller further from the anchor


def anchor_code_init(count: int, dim: int, generator: torch.Generator) -> Tensor:
    """Starting codes of ``count`` anchor objects: as perpendicular as ``dim`` dimensions allow.

    The first ``dim`` are a random orthonormal frame; the next ``dim`` are its negatives, so no
    two anchors start closer than a right angle and every direction of the code carries one.
    """
    if count > 2 * dim:
        raise ValueError(f"at most {2 * dim} anchors fit on perpendicular and opposite codes; got {count}")
    frame = torch.linalg.qr(torch.randn((dim, dim), generator=generator)).Q.T      # rows orthonormal
    return torch.cat([frame, -frame])[:count].contiguous()


def fit_rounded_boxes(grids: Tensor, generator: torch.Generator, points: int = 16384, steps: int = 800,
                      band: float = 0.05) -> typing.Tuple[Tensor, Tensor]:
    """Rounded box (b1, b2, b3, r) fitted to each of ``K`` stored fields (K×R×R×R, positive inside).

    Minimises the mean absolute difference between the box's field and the stored one at uniform
    points of the box, weighted by ``exp(-|target| / band)`` so the fit is decided near the surface.
    The box is axis-aligned and centred (the anchors are posed that way). Returns the K×4
    parameters and the K weighted mean errors.
    """
    from .decoder import rounded_box_field

    device = grids.device
    k = grids.shape[0]
    x = (torch.rand((1, points, 3), generator=generator) * 2.0 - 1.0).to(device)
    target = torch.cat([L.sample_grid(grids[s:s + 16].float(), x.expand(min(16, k - s), -1, -1))
                        for s in range(0, k, 16)])                                          # K x N
    weight = torch.exp(-target.abs() / band)
    weight = weight / weight.sum(dim=1, keepdim=True)
    # start: a sphere of the object's volume, extents a tenth
    volume = (target > 0).float().mean(dim=1) * 8.0
    start_r = (3.0 * volume / (4.0 * math.pi)).clamp_min(1e-3) ** (1.0 / 3.0)
    raw = torch.zeros((k, 4), device=device)
    raw[:, :3] = math.log(math.expm1(0.1))
    raw[:, 3] = torch.log(torch.expm1(start_r))
    raw.requires_grad_(True)
    optimiser = torch.optim.Adam([raw], lr=0.02)
    with torch.enable_grad():
        for _ in range(steps):
            base = torch.nn.functional.softplus(raw)
            error = ((rounded_box_field(x.expand(k, -1, -1), base) - target).abs() * weight).sum(dim=1)
            optimiser.zero_grad()
            error.sum().backward()
            optimiser.step()
    with torch.no_grad():
        base = torch.nn.functional.softplus(raw)
        error = ((rounded_box_field(x.expand(k, -1, -1), base) - target).abs() * weight).sum(dim=1)
    return base.detach(), error.detach()


def _first_rows(fb: FieldBatch, count: int) -> FieldBatch:
    """The first ``count`` codes of a field batch (slices, still in the autograd graph)."""
    return FieldBatch(codes=fb.codes[:count], cluster=fb.cluster[:count], x=fb.x[:count], f=fb.f[:count],
                      g=fb.g[:count], xp=fb.xp[:count], fp=fb.fp[:count], mod=None,
                      fp_weight=None if fb.fp_weight is None else fb.fp_weight[:count],
                      f_move=None if fb.f_move is None else fb.f_move[:count],
                      volume_weight=None if getattr(fb, "volume_weight", None) is None else fb.volume_weight[:count])


# ----------------------------------------------------------------------- trainer
class Trainer:
    """Model, optimiser, multipliers, anchor bank and the training step."""

    def __init__(
        self,
        cfg: DataFreeConfig,
        device: typing.Union[str, torch.device] = "cpu",
        output_dir: Optional[str] = None,
        tensorboard: bool = True,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.output_dir = output_dir
        torch.manual_seed(int(cfg.seed))
        # All sampling draws from this CPU generator, so a run is reproducible
        # and independent of the device.
        self.generator = torch.Generator().manual_seed(int(cfg.seed))
        self.model = ModulatedSphereDecoder(cfg).to(self.device)
        start = getattr(cfg, "init_main_from", None)
        if start:
            # A new run starts its main network from saved weights (a fitted sphere). The side
            # network keeps its own start.
            saved = torch.load(start, map_location=self.device)
            main = {key: value for key, value in saved.items() if key.startswith(("hidden.", "out."))}
            missing = self.model.load_state_dict(main, strict=False)
            if any(key.startswith(("hidden.", "out.")) for key in missing.missing_keys) or missing.unexpected_keys:
                raise ValueError(f"init_main_from does not match the main network: {missing}")
        self._warm_state: Optional[dict] = None
        if getattr(cfg, "init_model_from", None):
            self._warm_state = self._load_model_weights(str(cfg.init_model_from))
        self.optimizer = torch.optim.AdamW(
            self._param_groups(),
            lr=float(cfg.lr),
            betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
            weight_decay=float(cfg.weight_decay),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _Schedule(cfg))
        # Running average of the weights (does not take part in training).
        self.average: Optional[torch.nn.Module] = None
        if float(getattr(cfg, "weight_average_decay", 0.0)) > 0:
            self.average = self._copy_for_average()
        self.multipliers = AdaptiveMultipliers(cfg, names=CONSTRAINT_KEYS)
        self.bank = AnchorBank(cfg, self.generator, self.device)
        self.step_count = 0
        self.batch: Optional[Batch] = None
        self._metrics_file = None
        self._diag_file = None
        self._writer = None
        self._tensorboard = tensorboard
        # Weak-direction pairs: usage of each direction of code space, and an orthonormal basis
        # of the directions the decoder does not use (see measure_usage).
        self._usage: Optional[Tensor] = None
        self._weak_basis: Optional[Tensor] = None
        self._used_basis: Optional[Tensor] = None
        self._usage_info: Dict[str, float] = {}
        self._main_rows: Optional[int] = None
        if self._weak_on() and not (bool(cfg.batch_neighbours) and bool(cfg.global_codes)
                                    and str(cfg.diversity_mode) == "nearest" and int(cfg.accumulate) > 1):
            raise ValueError("weak_pairs and weak_frame_passes need global_codes, diversity_mode 'nearest', "
                             "batch_neighbours and accumulate > 1")
        if int(getattr(cfg, "weak_pairs", 0)) > 0 and self._frames() > 0:
            raise ValueError("weak_pairs and weak_frame_passes are two forms of one term: use one")
        # Anchor objects: target grids, one code each, and the optimiser of those codes.
        self.anchor_grid: Optional[Tensor] = None
        self.anchor_codes: Optional[torch.nn.Parameter] = None
        self.anchor_optimizer: Optional[torch.optim.Optimizer] = None
        if str(getattr(cfg, "anchor_path", "")):
            self._load_anchors()
        if self._warm_state is not None:
            self._load_warm_extras(self._warm_state)
            self._warm_state = None
        if self._budget_on():
            low_high = cfg.mult_ranges.get("base_deviation", (0.01, 1e5))
            self.multipliers.mult_ranges["base_deviation"] = (float(low_high[0]), float(low_high[1]))
            self.multipliers._mult["base_deviation"] = self.multipliers._clamp(
                float(cfg.base_budget_mult_init), "base_deviation")
            self.multipliers.tolerances["base_deviation"] = float(cfg.base_budget_start) ** 2
        if self._spread_on():
            low_high = cfg.mult_ranges.get("base_spread", (0.01, float(getattr(cfg, "base_spread_mult_max", 1e5))))
            self.multipliers.mult_ranges["base_spread"] = (float(low_high[0]), float(low_high[1]))
            self.multipliers._mult["base_spread"] = self.multipliers._clamp(float(cfg.base_spread_mult_init), "base_spread")
            self.multipliers.tolerances["base_spread"] = float(getattr(cfg, "base_spread_tolerance", 1e-6))
        self.anchor_base_fit: Optional[Dict[str, Any]] = None
        if bool(getattr(cfg, "anchor_base_fit", False)) and self.anchor_codes is not None:
            self._fit_anchor_bases()
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    @torch.no_grad()
    def _fit_anchor_bases(self) -> None:
        """Set the four base components of every anchor code so that its fixed-map base fits the object.

        The part of the code outside the four base directions keeps its direction and is rescaled
        so the code stays a unit vector. The fit, the base the map then gives (at full ramp) and
        the fit error are kept in ``self.anchor_base_fit`` and written to anchor_base_fit.json.
        """
        model = self.model
        gen = torch.Generator().manual_seed(int(self.cfg.seed) + 31337)
        target, error = fit_rounded_boxes(self.anchor_grid, gen)
        scores = model.scores_for_base(target)                                  # K x 4
        codes = torch.nn.functional.normalize(self.anchor_codes.detach(), dim=-1)
        proj = model.base_projection.to(codes.dtype)                            # D x 4
        dim = codes.shape[1]
        along = scores / math.sqrt(dim)                                         # wanted P^T z
        rest = codes - (codes @ proj) @ proj.T
        scale = torch.sqrt((1.0 - along.square().sum(-1)).clamp_min(1e-6)) / rest.norm(dim=-1).clamp_min(1e-9)
        self.anchor_codes.copy_(rest * scale[:, None] + along @ proj.T)
        was = float(model.base_range)
        model.base_range.fill_(1.0)
        got = model.prepare(self.anchor_codes.detach()).base
        model.base_range.fill_(was)
        self.anchor_base_fit = {
            "target": target.cpu().tolist(), "base": got.cpu().tolist(), "fit_error": error.cpu().tolist(),
            "scores": scores.cpu().tolist(),
            "code_change": float((self.anchor_codes.detach() - codes).norm(dim=-1).mean()),
        }
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "anchor_base_fit.json"), "w") as handle:
                json.dump(self.anchor_base_fit, handle)

    # ------------------------------------------------------------ warm start
    def _load_model_weights(self, path: str) -> dict:
        """Copy every matching weight of a saved model into this one (``init_model_from``).

        Weights whose input width grew (the base features appended to the main network's input)
        keep their old columns and the new ones stay at their start (zero). Weights the saved
        model lacks (the base head) keep their start. Any other mismatch is an error. Returns the
        saved trainer state (or ``{}`` for a plain state dict) for :meth:`_load_warm_extras`.
        """
        saved = torch.load(path, map_location=self.device, weights_only=False)
        state = saved if "model" in saved and isinstance(saved["model"], dict) else {"model": saved}
        weights = state["model"]
        own = self.model.state_dict()
        unknown = [key for key in weights if key not in own]
        if unknown:
            raise ValueError(f"init_model_from has weights this model lacks: {unknown[:8]}")
        with torch.no_grad():
            for key, value in weights.items():
                target = own[key]
                value = value.to(device=target.device, dtype=target.dtype)
                if value.shape == target.shape:
                    target.copy_(value)
                elif (value.dim() == 2 and target.dim() == 2 and value.shape[0] == target.shape[0]
                      and value.shape[1] < target.shape[1]):
                    target[:, : value.shape[1]].copy_(value)
                else:
                    raise ValueError(f"init_model_from: {key} has shape {tuple(value.shape)}, "
                                     f"this model {tuple(target.shape)}")
        return state

    def _load_warm_extras(self, state: dict) -> None:
        """Anchor codes and multipliers of the warm-start checkpoint, when they fit this run."""
        codes = state.get("anchor_codes")
        if self.anchor_codes is not None and codes is not None and codes.shape == self.anchor_codes.shape:
            with torch.no_grad():
                self.anchor_codes.copy_(codes.to(self.anchor_codes.device))
        if state.get("multipliers") is not None:
            self.multipliers.load_state_dict(state["multipliers"])

    # --------------------------------------------------------------- anchors
    def _load_anchors(self) -> None:
        cfg = self.cfg
        if not (bool(cfg.batch_neighbours) and bool(cfg.global_codes) and str(cfg.diversity_mode) == "nearest"
                and int(cfg.accumulate) > 1):
            raise ValueError("anchor objects need global_codes, diversity_mode 'nearest', batch_neighbours "
                             "and accumulate > 1")
        if int(getattr(cfg, "weak_pairs", 0)) > 0:
            raise ValueError("anchor objects are not combined with weak_pairs (a frame pass is fine)")
        data = torch.load(str(cfg.anchor_path), map_location="cpu", weights_only=False)
        posed = float(data.get("pose_power", 2.0))                      # files without the field: mass-based
        if abs(posed - float(cfg.pose_power)) > 1e-9:
            raise ValueError(f"the anchor objects in {cfg.anchor_path} were posed with pose_power {posed:g} but this "
                             f"run uses {float(cfg.pose_power):g}; the anchor pull would fight the pose terms. "
                             "Rebuild the anchors with the run's pose_power, or set pose_power to match.")
        grid = data["grid"]
        if not bool(data.get("positive_inside", True)):
            grid = -grid
        count = int(cfg.anchor_count) if int(cfg.anchor_count) > 0 else grid.shape[0]
        if count > grid.shape[0]:
            raise ValueError(f"anchor_count {count} but the file holds {grid.shape[0]} objects")
        self.anchor_grid = grid[:count].to(self.device)                 # K x R x R x R, kept in its stored precision
        gen = torch.Generator().manual_seed(int(cfg.seed) * 1000003 + 104729)
        codes = anchor_code_init(count, int(cfg.latent_dim), gen).to(self.device)
        self.anchor_codes = torch.nn.Parameter(codes)
        self.anchor_optimizer = torch.optim.Adam([self.anchor_codes], lr=0.0)

    def _consistency_tolerance(self, step: int) -> Optional[float]:
        """Scheduled consistency tolerance at ``step`` (None when there is no schedule)."""
        cfg = self.cfg
        start = getattr(cfg, "consistency_tolerance_start", None)
        if start is None:
            return None
        final = float(cfg.tolerances["consistency"])
        hold, ramp = int(cfg.consistency_hold_steps), int(cfg.consistency_tighten_steps)
        if step < hold:
            return float(start)
        if ramp <= 0 or step >= hold + ramp:
            return final
        return float(start) * (final / float(start)) ** ((step - hold) / ramp)     # geometric

    def _base_range(self, step: int) -> float:
        """Ramp factor of a rounded-box base at ``step``: min(1, step / base_ramp_steps)."""
        ramp = int(getattr(self.cfg, "base_ramp_steps", 0))
        return 1.0 if ramp <= 0 else min(1.0, step / ramp)

    def _budget_on(self) -> bool:
        return float(getattr(self.cfg, "base_budget", 0.0)) > 0.0

    def _base_budget(self, step: int) -> float:
        """Allowed RMS deviation of the base from the sphere at ``step`` (linear growth)."""
        cfg = self.cfg
        start, end, ramp = float(cfg.base_budget_start), float(cfg.base_budget), int(cfg.base_budget_steps)
        return end if ramp <= 0 or step >= ramp else start + (end - start) * step / ramp

    def _spread_on(self) -> bool:
        cfg = self.cfg
        return any(float(getattr(cfg, name, 0.0)) > 0 for name in (
            "base_spread_target", "base_spread_target_r", "base_spread_target_bend", "base_spread_target_taper"))

    def _base_spread(self, fb) -> typing.Tuple[Tensor, Dict[str, float]]:
        """Shortfall of the spread of the base parameters across the pass's codes (see base_spread_target)."""
        cfg = self.cfg
        base = fb.mod.base
        targets = [float(cfg.base_spread_target)] * 3 + [float(cfg.base_spread_target_r)]
        deform = getattr(fb.mod, "deform", None)
        if deform is not None:
            base = torch.cat([base, deform[:, :9]], dim=-1)
            ramp = int(getattr(cfg, "base_spread_ramp_steps", 0))
            grow = 1.0 if ramp <= 0 else min(max((self.step_count - int(cfg.base_spread_ramp_from)) / ramp, 0.0), 1.0)
            targets += ([grow * float(getattr(cfg, "base_spread_target_bend", 0.0))] * 6
                        + [grow * float(getattr(cfg, "base_spread_target_taper", 0.0))] * 3)
        std = base.std(dim=0) if base.shape[0] > 1 else torch.zeros_like(base[0])
        target = torch.tensor(targets, device=base.device, dtype=base.dtype)
        value = torch.relu(target - std).square().sum()
        info = {"base/spread_shortfall": float(value.detach()), "base/std_b": float(std[:3].mean().detach()),
                "base/std_r": float(std[3].detach())}
        if deform is not None:
            info["base/std_bend"] = float(std[4:10].mean().detach())
            info["base/std_taper"] = float(std[10:13].mean().detach())
        return value, info

    def _correction_cost(self, fb, rows: Optional[int] = None) -> typing.Tuple[Tensor, Dict[str, float]]:
        """Mean squared correction ``f - base(x)`` at the sample points within the band of the surface.

        ``rows``: only the first ``rows`` codes (the anchor rows at the end are left out). The band is
        taken from the field's value, detached, so the cost does not move points in or out of it.
        """
        f = fb.f if rows is None else fb.f[:rows]
        x = fb.x if rows is None else fb.x[:rows]
        mod = fb.mod if rows is None else fb.mod.index(torch.arange(rows, device=f.device))
        correction = f - self.model.base_field(mod, x)
        near = (f.detach().abs() < float(self.cfg.correction_cost_band)).to(f.dtype)
        value = (correction.square() * near).sum() / near.sum().clamp_min(1.0)
        with torch.no_grad():
            info = {"corr/near_sq": float(value), "corr/near_mean": float((correction * near).sum() / near.sum().clamp_min(1.0)),
                    "corr/near_share": float(near.mean())}
        return value, info

    def _repel_on(self) -> bool:
        return float(getattr(self.cfg, "base_repel_weight", 0.0)) > 0 and getattr(self.model, "box_base", False)

    def _base_vector(self, mod) -> Tensor:
        """Scaled base vector of each code: extents and radius, then bends and tapers when present."""
        cfg = self.cfg
        parts = [mod.base / float(cfg.base_repel_scale_shape)]
        if getattr(mod, "deform", None) is not None:
            parts += [mod.deform[:, :6] / float(cfg.base_repel_scale_bend),
                      mod.deform[:, 6:9] / float(cfg.base_repel_scale_taper)]
        return torch.cat(parts, dim=-1)

    def _field_grad_scales(self) -> Dict[str, float]:
        cfg = self.cfg
        scales = {"eikonal": float(getattr(cfg, "base_eikonal_grad_scale", 1.0)),
                  "consistency": float(getattr(cfg, "base_consistency_grad_scale", 1.0))}
        return {k: v for k, v in scales.items() if v != 1.0}

    def _backward(self, loss: Tensor, fb, constraints: Dict[str, Tensor], divide: float) -> Dict[str, float]:
        """``(loss / divide).backward()``, with the field terms' gradient into the base scaled if asked.

        The slope and consistency terms' gradient with respect to the base parameters (``fb.mod.base``)
        is taken once on its own; the backward pass then adds ``(scale - 1)`` times it through the base
        alone, so the base gets ``scale`` times that part and everything else is exact.
        """
        scales = self._field_grad_scales()
        base = getattr(fb.mod, "base", None) if fb.mod is not None else None
        if not scales or base is None or not base.requires_grad:
            (loss / divide).backward()
            return {}
        part = sum((scale - 1.0) * self.multipliers.multiplier(k) * constraints[k] for k, scale in scales.items())
        (g,) = torch.autograd.grad(part / divide, base, retain_graph=True, allow_unused=True)
        if g is None:
            (loss / divide).backward()
            return {}
        torch.autograd.backward([loss / divide, base], [None, g])
        return {"base/field_grad_change": float(g.norm())}

    def _base_terms(self, fb) -> typing.Tuple[Tensor, Tensor, Dict[str, float]]:
        """Deviation of the base from the sphere (mean over codes of |b|^2 + (r - R)^2) and the
        negative-extent cost (mean of relu(-b)^2 summed over axes), from the pass's codes."""
        base = fb.mod.base
        deviation = torch.cat([base[:, :3], base[:, 3:] - float(self.cfg.base_radius)], dim=-1)
        value = deviation.square().sum(-1).mean()
        negative = torch.relu(-base[:, :3]).square().sum(-1).mean()
        with torch.no_grad():
            info = {"base/deviation": float(value), "base/negative": float(negative),
                    "base/spread": float(base.std(dim=0).mean()) if base.shape[0] > 1 else 0.0,
                    "base/b_mean": float(base[:, :3].mean()), "base/r_mean_batch": float(base[:, 3].mean())}
        return value, negative, info

    def _apply_schedules(self, step: int) -> None:
        if self._budget_on():
            self.multipliers.tolerances["base_deviation"] = self._base_budget(step) ** 2
        if getattr(self.model, "box_base", False):
            for model in (self.model, self.average):
                if model is not None:
                    model.base_range.fill_(self._base_range(step))
        tolerance = self._consistency_tolerance(step)
        if tolerance is not None:
            self.multipliers.tolerances["consistency"] = tolerance
        for name, top in self._field_limits(step).items():
            low = self.cfg.mult_ranges.get(name, (self.cfg.mult_min, self.cfg.mult_max))[0]
            self.multipliers.mult_ranges[name] = (min(float(low), top), top)
            self.multipliers._mult[name] = self.multipliers._clamp(self.multipliers.multiplier(name), name)

    def _variety_weight_now(self) -> float:
        """Weight of the variety objective: the configured one, or the controller's current value."""
        cfg = self.cfg
        base, top = float(getattr(cfg, "variety_weight", 1.0)), float(getattr(cfg, "variety_weight_max", 0.0))
        if top <= 0:
            return base
        value = getattr(self, "_variety_weight", None)
        return base if value is None else min(max(float(value), base), top)

    def _adapt_variety_weight(self, measured: bool) -> bool:
        """Move the variety weight one update (see ``variety_weight_max``); returns whether it was safe to grow."""
        cfg, mult = self.cfg, self.multipliers
        consistency = mult.ema("consistency")
        safe = (measured and consistency is not None
                and consistency < float(cfg.variety_consistency_ceiling)
                and not any(mult.violated(name) for name in ("bridge", "delete", "fill", "widen")))
        factor = 1.0 + float(cfg.variety_weight_up) if safe else 1.0 - float(cfg.variety_weight_down)
        self._variety_weight = self._variety_weight_now() * factor
        self._variety_weight = self._variety_weight_now()           # clamped to the configured range
        return bool(safe)

    def _field_limits(self, step: int) -> Dict[str, float]:
        """Upper limits of the eikonal and consistency multipliers at ``step`` (empty without a ramp).

        Each limit rises geometrically from ``field_ramp_start`` to its configured value over
        ``field_ramp_steps`` updates, so the two terms weigh next to nothing at first and are back
        at full strength when the ramp ends.
        """
        cfg = self.cfg
        limits = {}
        ramp = int(getattr(cfg, "field_ramp_steps", 0))
        if ramp > 0:
            start = float(cfg.field_ramp_start)
            for name in ("eikonal", "consistency"):
                high = float(cfg.mult_ranges.get(name, (cfg.mult_min, cfg.mult_max))[1])
                limits[name] = high if step >= ramp else min(high, start * (high / start) ** (step / ramp))
        cap = float(getattr(cfg, "field_cap", 0.0) or 0.0)
        if cap > 0:
            until = int(getattr(cfg, "field_cap_until", 0))
            release = max(int(getattr(cfg, "field_cap_release_steps", 0)), 0)
            target = float(getattr(cfg, "field_cap_release_to", 100.0))
            for name in tuple(getattr(cfg, "field_cap_names", ("eikonal",))):
                high = float(cfg.mult_ranges.get(name, (cfg.mult_min, cfg.mult_max))[1])
                if step < until:
                    top = cap
                elif step < until + release:
                    top = cap * (target / cap) ** ((step - until) / release)
                else:
                    continue                                   # released: the configured limit
                limits[name] = min(limits.get(name, high), top)
        return limits

    def _anchor_weight(self, step: int) -> float:
        """Weight of the anchor pull at ``step`` (a linear ramp when anchor_weight_ramp_steps > 0)."""
        cfg = self.cfg
        top, ramp = float(cfg.anchor_weight), int(getattr(cfg, "anchor_weight_ramp_steps", 0))
        hold = max(int(getattr(cfg, "anchor_weight_hold_steps", 0)), 0)
        if (ramp <= 0 and hold <= 0) or step >= hold + ramp:
            return top
        low = float(getattr(cfg, "anchor_weight_start", 0.01))
        return top * (low + (1.0 - low) * max(step - hold, 0) / max(ramp, 1))

    def _anchor_lr(self, step: int) -> float:
        """Learning rate of the anchor codes: zero while frozen, then a linear ramp."""
        cfg = self.cfg
        top, freeze = float(cfg.anchor_embed_lr), int(cfg.anchor_freeze_steps)
        if self.anchor_codes is None or top <= 0 or step < freeze:
            return 0.0
        return top * min(1.0, (step - freeze + 1) / max(1, int(cfg.anchor_ramp_steps)))

    # ------------------------------------------------------ weak directions
    def _frames(self) -> int:
        return min(max(0, int(getattr(self.cfg, "weak_frame_passes", 0))), int(self.cfg.accumulate))

    def _weak_on(self) -> bool:
        return int(getattr(self.cfg, "weak_pairs", 0)) > 0 or self._frames() > 0

    def _needs_usage(self) -> bool:
        """Whether the per-direction usage of code space has to be measured (not for a full frame)."""
        return int(getattr(self.cfg, "weak_pairs", 0)) > 0 or (
            self._frames() > 0 and not bool(getattr(self.cfg, "weak_frame_full", False)))

    def measure_usage(self) -> Tensor:
        """How much each direction of code space changes the field, as a ``D x D`` matrix.

        For ``weak_probe_codes`` random codes the field is evaluated, without parameter
        gradients, at ``weak_probe_points`` points for the code moved ``weak_chord`` each way
        along every tangent direction. The pose-corrected change per unit chord gives the mean
        squared field change per direction; it is rotated into the coordinates of code space
        and averaged over the probe codes. Eigenvectors with small eigenvalues are directions
        the decoder ignores. It draws from its own generator (seeded by the run seed and the
        update count), so the training sample stream is not disturbed.
        """
        cfg, model, device = self.cfg, self.model, self.device
        dim = int(cfg.latent_dim)
        gen = torch.Generator().manual_seed(int(cfg.seed) * 1000003 + 7919 + int(self.step_count))
        count, points = int(cfg.weak_probe_codes), int(cfg.weak_probe_points)
        codes = sample_sphere(count, dim, gen, device)
        theta = 2.0 * math.asin(min(1.0, 0.5 * float(cfg.weak_chord)))
        span = 2.0 * math.sin(theta)                                   # chord between the two moved codes
        rows = dim - 1
        usage = torch.zeros((dim, dim), dtype=torch.float64)
        for z in codes:
            x = (torch.rand((1, points, 3), generator=gen) * 2.0 - 1.0).to(device)
            mask = (x.square().sum(dim=-1) < float(cfg.bound_radius) ** 2).to(x.dtype)
            f0, g0 = field_and_grad(model, model.prepare(z[None]), x, create_graph=False)
            f0, g0 = f0.detach(), g0.detach()
            frame = tangent_basis(z)                                   # (D-1) x D
            moved = torch.cat([math.cos(theta) * z + math.sin(theta) * frame,
                               math.cos(theta) * z - math.sin(theta) * frame])
            with torch.no_grad():
                values = torch.cat([model(moved[s:s + 64], x.expand(moved[s:s + 64].shape[0], -1, -1))
                                    for s in range(0, moved.shape[0], 64)])
                delta = (values[:rows] - values[rows:]) / span
                if bool(getattr(cfg, "diversity_pose_invariant", False)):
                    delta = L._remove_pose(delta, x.expand(rows, -1, -1), g0.expand(rows, -1, -1),
                                           f0.expand(rows, -1), mask.expand(rows, -1))
                delta = delta * mask
                local = (delta @ delta.T) / mask.sum().clamp_min(1.0)  # (D-1) x (D-1)
                usage += (frame.T @ local @ frame).double().cpu() / count
        self._set_usage(usage)
        return self._usage

    def _set_usage(self, usage: Tensor) -> None:
        cfg = self.cfg
        usage = usage.detach().double().cpu()
        usage = 0.5 * (usage + usage.T)
        self._usage = usage
        lam, vec = torch.linalg.eigh(usage)
        lam = lam.clamp_min(0.0)
        # Unused directions: field change per unit chord (RMS) at or below the floor. On a unit
        # sphere a step along an unused axis also rescales the used part of the code, so these
        # eigenvalues are small but not zero; the floor has to sit above that level.
        weak = lam.sqrt() <= float(cfg.weak_floor)
        self._weak_basis = vec[:, weak].to(device=self.device, dtype=torch.float32)     # D x m, orthonormal
        self._used_basis = vec[:, ~weak].flip(1).to(device=self.device, dtype=torch.float32)   # strongest first
        share = lam / lam.sum().clamp_min(1e-30)
        self._usage_info = {
            "weak/usage_effective": float(torch.exp(-(share * torch.log(share + 1e-30)).sum())),
            "weak/usage_top": float(lam.max().sqrt()),
            "weak/usage_median": float(lam.median().sqrt()),
            "weak/used_directions": float((~weak).sum()),
        }

    # -------------------------------------------------------------- sampling
    def _frame_batch(self, total: int) -> Batch:
        """A pass whose first code has a frame of partners (see ``cfg.weak_frame_passes``).

        The last rows are the partners of code 0, each ``weak_chord`` away: first one along every
        used direction of code space (a plain step on the sphere), then ``weak_frame_new`` with
        the used part kept and the unused part turned, in mutually perpendicular directions.
        """
        cfg, gen, device = self.cfg, self.generator, self.device
        if bool(getattr(cfg, "weak_frame_full", False)):
            # The complete frame: a partner along every tangent direction at code 0. The
            # log-volume of a full frame does not depend on which perpendicular directions are
            # used, so no direction usage has to be measured.
            partners = min(int(cfg.latent_dim) - 1, total - 1)
            codes = sample_sphere(total - partners, cfg.latent_dim, gen, device)
            centre = codes[0]
            theta = 2.0 * math.asin(min(1.0, 0.5 * float(cfg.weak_chord)))
            partner = math.cos(theta) * centre + math.sin(theta) * tangent_basis(centre)[:partners]
            partner = torch.nn.functional.normalize(partner, dim=-1)
            return Batch(codes=torch.cat([codes, partner]),
                         cluster=torch.zeros(total, dtype=torch.long, device=device),
                         x=sample_points(cfg, 1, gen, device),
                         bank_index=torch.full((1,), -1, dtype=torch.long, device=device),
                         eps=torch.full((1,), 2.0 ** 0.5, device=device),
                         pair_first=torch.zeros(partners, dtype=torch.long, device=device),
                         pair_second=torch.arange(total - partners, total, device=device),
                         pair_chord=(centre - partner).norm(dim=-1), frame_used=partners)
        if self._weak_basis is None:
            self.measure_usage()
        used, unused = self._used_basis, self._weak_basis               # D x k, D x m
        new = max(0, min(int(cfg.weak_frame_new), unused.shape[1] - 1))
        keep = max(0, min(used.shape[1], total - 1 - new))              # the strongest that fit
        partners = keep + new
        codes = sample_sphere(total - partners, cfg.latent_dim, gen, device)
        centre = codes[0]
        chord = float(cfg.weak_chord)
        parts = []
        if keep > 0:
            axes = used[:, :keep].T                                     # k x D
            axes = torch.nn.functional.normalize(axes - (axes @ centre)[:, None] * centre, dim=-1)
            theta = 2.0 * math.asin(min(1.0, 0.5 * chord))
            parts.append(math.cos(theta) * centre + math.sin(theta) * axes)
        if new > 0:
            inside = (centre @ unused) @ unused.T                        # the centre's unused part
            size = inside.norm().clamp_min(1e-6)
            noise = torch.randn((new, unused.shape[1]), generator=gen).to(device) @ unused.T
            noise = noise - (noise @ inside)[:, None] * inside / size.square()
            turn = torch.linalg.qr(noise.T).Q.T                          # new x D, orthonormal, inside the unused set
            angle = 2.0 * torch.asin((0.5 * chord / size).clamp(max=1.0))
            parts.append((centre - inside) + torch.cos(angle) * inside + torch.sin(angle) * size * turn)
        first = second = lengths = None
        if partners > 0:
            partner = torch.nn.functional.normalize(torch.cat(parts), dim=-1)
            lengths = (centre - partner).norm(dim=-1)
            codes = torch.cat([codes, partner])
            first = torch.zeros(partners, dtype=torch.long, device=device)
            second = torch.arange(total - partners, total, device=device)
        cluster = torch.zeros(total, dtype=torch.long, device=device)
        x = sample_points(cfg, 1, gen, device)
        return Batch(codes=codes, cluster=cluster, x=x,
                     bank_index=torch.full((1,), -1, dtype=torch.long, device=device),
                     eps=torch.full((1,), 2.0 ** 0.5, device=device),
                     pair_first=first, pair_second=second, pair_chord=lengths,
                     frame_used=keep if partners > 0 else -1)

    def sample_batch(self, frame: bool = False, anchors: Optional[Tensor] = None) -> Batch:
        """New clusters (part from the bank, part fresh) and their shared points.

        With a rounded-box base and ``base_surface_points`` some rows of the points are moved
        next to base surfaces (see :meth:`_base_surface_rows`).
        """
        batch = self._sample_batch(frame=frame, anchors=anchors)
        if getattr(self.model, "box_base", False) and int(getattr(self.cfg, "base_surface_points", 0)) > 0:
            self._base_surface_rows(batch)
        return batch

    @torch.no_grad()
    def _base_surface_rows(self, batch: Batch) -> None:
        """Write points near base surfaces over rows of ``batch.x`` (in place).

        Cluster mode: each cluster's rows go next to the base of its first code, just before the
        hard-negative rows. Shared points (one row of points for the whole batch): the points are
        split between up to 8 of the batch's codes, taken at random. A point is a random point of
        the core box moved out by r in a random direction, projected onto the surface by one exact
        step ``p - f(p) grad f(p)``, then offset along the normal by N(0, base_surface_noise).
        """
        from .decoder import rounded_box_field

        cfg, gen, x = self.cfg, self.generator, batch.x
        rows, n = x.shape[0], x.shape[1]
        reserved = int(cfg.hard_negatives) if rows > 1 else 0
        count = min(int(cfg.base_surface_points), n - 1 - reserved)
        if count <= 0:
            return
        if rows > 1:
            first = torch.zeros(rows, dtype=torch.long, device=x.device)
            first = first.scatter_reduce(0, batch.cluster, torch.arange(batch.cluster.numel(), device=x.device),
                                         reduce="amin", include_self=False)
            owner = first[:, None].expand(rows, count)                                  # rows x count
        else:
            pick = torch.randperm(batch.codes.shape[0], generator=gen)[: min(8, batch.codes.shape[0])]
            owner = pick.to(x.device)[torch.arange(count, device=x.device) % pick.numel()][None]
        base = self.model.prepare(batch.codes).base.detach()[owner.reshape(-1)]         # K x 4
        k = base.shape[0]
        core = base[:, :3].clamp_min(0.0)
        centre = (torch.rand((k, 3), generator=gen).to(x.device) * 2.0 - 1.0) * core
        direction = torch.nn.functional.normalize(torch.randn((k, 3), generator=gen).to(x.device), dim=-1)
        p = (centre + base[:, 3:] * direction)[:, None, :]                              # K x 1 x 3
        with torch.enable_grad():
            p = p.requires_grad_(True)
            f = rounded_box_field(p, base)
            (g,) = torch.autograd.grad(f.sum(), p)
        surface = (p - f[..., None] * g).detach()
        noise = torch.randn((k, 1, 1), generator=gen).to(x.device) * float(cfg.base_surface_noise)
        points = (surface + noise * g).reshape(rows, count, 3).clamp(-1.0, 1.0)
        start = n - reserved - count
        x[:, start: start + count] = points.to(x.dtype)

    def _sample_batch(self, frame: bool = False, anchors: Optional[Tensor] = None) -> Batch:
        cfg, gen, device = self.cfg, self.generator, self.device
        clusters, per = int(cfg.clusters), int(cfg.codes_per_cluster)
        if bool(getattr(cfg, "global_codes", False)):
            # No clusters: clusters x codes_per_cluster independent random codes that all share
            # one set of sample points, so every pair can be compared. No anchor bank, hence no
            # hard-negative points, in this mode.
            total = clusters * per
            if frame and self._frames() > 0:
                return self._frame_batch(total)
            if anchors is not None and anchors.numel() > 0:
                # The last rows are the codes of the chosen anchor objects (in the autograd graph
                # of the code table only while that table is being trained).
                own = torch.nn.functional.normalize(self.anchor_codes[anchors], dim=-1)
                if self._anchor_lr(self.step_count) <= 0:
                    own = own.detach()
                free = sample_sphere(total - anchors.numel(), cfg.latent_dim, gen, device)
                pull = None
                radius = float(getattr(cfg, "anchor_radius", 0.0))
                if radius > 0:
                    # A neighbourhood instead of a single code: each anchor is met at a random
                    # code up to two radii away (uniform in distance, so the centre is seen as
                    # often as the rim), and pulled with a weight that falls off with distance.
                    count = anchors.numel()
                    away = (2.0 * radius * torch.rand(count, generator=gen)).to(device).clamp(max=1.9)
                    side = torch.randn((count, int(cfg.latent_dim)), generator=gen).to(device)
                    side = side - (side * own.detach()).sum(dim=-1, keepdim=True) * own.detach()
                    side = torch.nn.functional.normalize(side, dim=-1)
                    angle = (2.0 * torch.asin(0.5 * away)).unsqueeze(-1)
                    own = torch.cos(angle) * own + torch.sin(angle) * side
                    pull = torch.exp(-(away / radius).square())
                return Batch(codes=torch.cat([free, own]), cluster=torch.zeros(total, dtype=torch.long, device=device),
                             x=sample_points(cfg, 1, gen, device),
                             bank_index=torch.full((1,), -1, dtype=torch.long, device=device),
                             eps=torch.full((1,), 2.0 ** 0.5, device=device), anchor_index=anchors,
                             anchor_pull=pull)
            pairs = min(int(getattr(cfg, "weak_pairs", 0)), total // 2)
            if pairs > 0:
                if self._weak_basis is None:
                    self.measure_usage()
                if self._weak_basis.shape[1] < 2:                       # nothing unused to rotate in
                    pairs = 0
            codes = sample_sphere(total - pairs, cfg.latent_dim, gen, device)
            first = second = lengths = None
            if pairs > 0:
                # The last `pairs` codes are partners of the first `pairs`. A partner keeps its
                # base code's used part exactly and has the unused part turned, inside the unused
                # directions, far enough to sit weak_chord away. (A plain step along an unused
                # axis would not do: staying on the unit sphere rescales the used part too.)
                basis = self._weak_basis                                 # D x m
                base = codes[:pairs]
                inside = (base @ basis) @ basis.T                        # the base's unused part
                size = inside.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                noise = torch.randn((pairs, basis.shape[1]), generator=gen).to(device) @ basis.T
                turn = noise - (noise * inside).sum(dim=-1, keepdim=True) * inside / size.square()
                turn = torch.nn.functional.normalize(turn, dim=-1)
                angle = 2.0 * torch.asin((0.5 * float(cfg.weak_chord) / size).clamp(max=1.0))
                partner = (base - inside) + torch.cos(angle) * inside + torch.sin(angle) * size * turn
                partner = torch.nn.functional.normalize(partner, dim=-1)
                lengths = (base - partner).norm(dim=-1)
                codes = torch.cat([codes, partner])
                first = torch.arange(pairs, device=device)
                second = torch.arange(total - pairs, total, device=device)
            cluster = torch.zeros(total, dtype=torch.long, device=device)
            x = sample_points(cfg, 1, gen, device)
            return Batch(codes=codes, cluster=cluster, x=x,
                         bank_index=torch.full((1,), -1, dtype=torch.long, device=device),
                         eps=torch.full((1,), 2.0 ** 0.5, device=device),
                         pair_first=first, pair_second=second, pair_chord=lengths)
        from_bank = min(int(round(float(cfg.bank_fraction) * clusters)), clusters, self.bank.size)
        bank_index = torch.full((clusters,), -1, dtype=torch.long)
        bank_index[:from_bank] = torch.randperm(self.bank.size, generator=gen)[:from_bank]
        bank_index = bank_index.to(device)
        centres = sample_sphere(clusters, cfg.latent_dim, gen, device)
        centres[:from_bank] = self.bank.codes[bank_index[:from_bank]]
        low, high = math.log(float(cfg.eps_min)), math.log(float(cfg.eps_max))
        eps = torch.exp(low + (high - low) * torch.rand(clusters, generator=gen)).to(device)
        codes = sample_cluster(centres, per, eps, gen).reshape(clusters * per, cfg.latent_dim)
        cluster = torch.arange(clusters, device=device).repeat_interleave(per)
        extra = None
        if from_bank > 0 and int(cfg.hard_negatives) > 0:
            extra = torch.full((clusters, int(cfg.hard_negatives), 3), float("nan"), device=device)
            extra[:from_bank] = self.bank.read(bank_index[:from_bank])
        x = sample_points(cfg, clusters, gen, device, extra=extra)
        return Batch(codes=codes, cluster=cluster, x=x, bank_index=bank_index, eps=eps)

    # ------------------------------------------------------------------ step
    def _sync(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _store_hard_negatives(self, batch: Batch, graph: ShapeGraph, fb) -> None:
        capacity = int(self.cfg.hard_negatives)
        is_bank = batch.bank_index >= 0
        if capacity <= 0 or not bool(is_bank.any()):
            return
        found = hard_negatives(graph, fb, self.cfg, capacity, generator=self.generator)
        centre_rows = torch.arange(batch.x.shape[0], device=self.device) * int(self.cfg.codes_per_cluster)
        self.bank.write(batch.bank_index[is_bank], found[centre_rows][is_bank])

    def _diversity(self, fb, step: int) -> typing.Tuple[Tensor, Dict[str, Tensor]]:
        """The diversity loss and its info; with a pool, the neighbour search runs over the pool too.

        In pooled mode the in-batch all-pairs pass (``L.diversity``) is skipped: the loss and every
        logged value come from ``L.nearest_pool``, measured over the chosen neighbour pairs.
        """
        cfg = self.cfg
        shared = getattr(self, "_shared_neighbours", None)
        if shared is not None:
            # Batch-wide search: the other passes of this update (and an optional pool), all
            # evaluated once at the points the passes share. See _accumulated_step.
            pool_f, pool_codes, index = shared
            rows = getattr(self, "_main_rows", None)                   # weak-direction partners stay out
            main = fb if rows is None else _first_rows(fb, rows)
            diversity, info = L.nearest_pool(main, pool_f, index, cfg, pool_codes=pool_codes, pool_terms=True,
                                             norm=getattr(self, "_main_norm", None))
            self._pool_share = float(info["pool_neighbour_share"])
            return diversity, info
        pool = int(getattr(cfg, "pool_size", 0))
        if pool <= 0 or not bool(getattr(cfg, "global_codes", False)) or str(cfg.diversity_mode) != "nearest":
            return L.diversity(fb, cfg, step)
        count = fb.x.shape[1]
        take = max(1, min(int(cfg.pool_points), count - 1))
        index = (1 + torch.randperm(count - 1, generator=self.generator)[:take]).to(self.device)   # skip the origin row
        codes = sample_sphere(pool, cfg.latent_dim, self.generator, self.device)
        x = fb.x[:1, index].detach()
        with torch.no_grad():
            pool_f = torch.cat([self.model(codes[s:s + 64], x.expand(codes[s:s + 64].shape[0], -1, -1))
                                for s in range(0, pool, 64)])
        diversity, info = L.nearest_pool(fb, pool_f, index, cfg, pool_codes=codes)
        self._pool_share = float(info["pool_neighbour_share"])
        return diversity, info

    def _objective(self, fb, diversity: Tensor) -> typing.Tuple[Tensor, float, float]:
        """Diversity plus the weighted spread term; also the spread value and mean |correlation|."""
        objective = diversity
        self._stat_spreads = {}
        push = float(getattr(self.cfg, "stat_repulsion_weight", 0.0))
        if push > 0:
            value, info = L.statistic_repulsion(fb, self.cfg)
            objective = objective + push * value
            self._stat_spreads = {k: float(v) for k, v in info.items()}
        local = float(getattr(self.cfg, "local_chamfer_weight", 0.0))
        if local > 0:
            value, info = L.local_chamfer(fb, self.cfg, generator=self.generator)
            objective = objective + local * value
            self._stat_spreads.update({k: float(v) for k, v in info.items()})
        weight = float(getattr(self.cfg, "spread_weight", 0.0))
        if weight <= 0:
            return objective, 0.0, 0.0
        value, info = L.spread(fb, self.cfg)
        return objective + weight * value, float(value.detach()), float(info["corr_abs_mean"])

    def _accumulated_step(self) -> Dict[str, Any]:
        """One optimiser step whose gradient is the mean over ``cfg.accumulate`` fresh batches.

        Each sub-batch has its own codes, points and graph (no graph reuse: more distinct shapes
        per update is the point). Multipliers are held fixed across the sub-batches and updated
        once, from the mean constraint values. Clipping is applied to the accumulated gradient.
        """
        cfg, model = self.cfg, self.model
        parts = int(cfg.accumulate)
        step = self.step_count
        self._apply_schedules(step)
        model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        sums: Dict[str, float] = {}
        modes, finite = [], True
        t0 = self._sync()

        def add(key: str, value) -> None:
            sums[key] = sums.get(key, 0.0) + float(value) / parts

        # Batch-wide neighbour search: every shape of the update is compared with all the others.
        # The passes share a block of sample points; all codes are evaluated there once without
        # gradients, and each pass then treats the other passes' shapes as fixed neighbours. With
        # no optimiser step in between, the gradients summed over the passes equal the gradient
        # of the nearest-neighbour loss over the whole update, at the memory cost of one pass.
        wide = (bool(getattr(cfg, "batch_neighbours", False)) and bool(getattr(cfg, "global_codes", False))
                and str(cfg.diversity_mode) == "nearest")
        batches, all_f, all_codes, shared_index, extra_f, extra_codes = None, None, None, None, None, None
        mains: List[int] = []
        if wide:
            if self._needs_usage() and (self._weak_basis is None
                                        or step % max(1, int(cfg.weak_refresh_every)) == 0):
                self.measure_usage()
            frames = self._frames()
            if self.anchor_codes is not None and int(cfg.anchor_codes_per_pass) > 0 and frames < parts:
                # A different subset of the anchor objects every update, split over the passes
                # that do not carry a frame.
                per_pass = min(int(cfg.anchor_codes_per_pass), int(cfg.clusters) * int(cfg.codes_per_cluster) - 2)
                known, need = self.anchor_codes.shape[0], per_pass * (parts - frames)
                order = torch.cat([torch.randperm(known, generator=self.generator)
                                   for _ in range(-(-need // known))])[:need].to(self.device)
                batches = [self.sample_batch(frame=True) if i < frames else
                           self.sample_batch(anchors=order[(i - frames) * per_pass:(i - frames + 1) * per_pass])
                           for i in range(parts)]
                self.anchor_optimizer.zero_grad(set_to_none=True)
            else:
                batches = [self.sample_batch(frame=True) if i < frames else self.sample_batch()
                           for i in range(parts)]
            count = batches[0].x.shape[1]
            take = max(1, min(int(cfg.pool_points), count - 1))
            shared_index = torch.arange(1, 1 + take, device=self.device)
            common = batches[0].x[:, 1:1 + take].clone()
            for b in batches[1:]:
                b.x[:, 1:1 + take] = common                             # rows 1..take are shared
            # Weak-direction partners and anchor codes (the last rows of a pass) are not neighbours
            # of anything.
            mains = [b.codes.shape[0] - (0 if b.pair_second is None else int(b.pair_second.numel()))
                     - (0 if b.anchor_index is None else int(b.anchor_index.numel())) for b in batches]
            all_codes = torch.cat([b.codes[:n].detach() for b, n in zip(batches, mains)])
            # passes of unequal size are weighted by the mean size (see losses.nearest_pool)
            self._main_norm = sum(mains) / parts if len(set(mains)) > 1 else None
            pool = int(getattr(cfg, "pool_size", 0))
            with torch.no_grad():
                was_training = model.training
                all_f = torch.cat([model(all_codes[s:s + 64], common.expand(all_codes[s:s + 64].shape[0], -1, -1))
                                   for s in range(0, all_codes.shape[0], 64)])
                all_vec = None
                if self._repel_on():
                    all_vec = torch.cat([self._base_vector(model.prepare(all_codes[s:s + 256]))
                                         for s in range(0, all_codes.shape[0], 256)])
                if pool > 0:                                            # optional extra crowd, once per update
                    extra_codes = sample_sphere(pool, cfg.latent_dim, self.generator, self.device)
                    extra_f = torch.cat([model(extra_codes[s:s + 64], common.expand(extra_codes[s:s + 64].shape[0], -1, -1))
                                         for s in range(0, pool, 64)])
                model.train(was_training)

        for part in range(parts):
            if wide:
                batch = batches[part]
                size, start = mains[part], sum(mains[:part])
                self._main_rows = size if size < batch.codes.shape[0] else None
                rest = torch.ones(all_codes.shape[0], dtype=torch.bool, device=self.device)
                rest[start:start + size] = False
                others_f, others_codes = all_f[rest], all_codes[rest]
                if extra_f is not None:
                    others_f, others_codes = torch.cat([others_f, extra_f]), torch.cat([others_codes, extra_codes])
                self._shared_neighbours = (others_f, others_codes, shared_index)
                self._shared_base = all_vec[rest] if all_vec is not None else None
            else:
                batch = self.sample_batch()
            ta = self._sync()
            fb = evaluate_field(model, batch.codes, batch.cluster, batch.x, cfg)
            tb = self._sync()
            e = float(L.consistency_error(fb))
            mode = graph_mode_for(e, cfg)
            graph = None
            tc = td = tb
            if mode != "off":
                graph = build_graph(fb, e, cfg, mode=mode)
                tc = self._sync()
                graph = confirm(model, fb.mod, graph, cfg)
                self._store_hard_negatives(batch, graph, fb)
                td = self._sync()
            modes.append(mode)
            diversity, info = self._diversity(fb, step)
            # The two field-accuracy constraints can leave the anchor rows out: those rows are tied
            # to exact distance values by the anchor pull, and real objects with sharp edges carry
            # about twice the error of the decoder's own shapes, which otherwise holds the
            # multiplier at its cap and makes the random codes pay for it.
            exact = fb
            if batch.anchor_index is not None and bool(getattr(cfg, "anchor_exempt_field", False)):
                exact = _first_rows(fb, batch.codes.shape[0] - int(batch.anchor_index.numel()))
            sloped = exact
            if bool(getattr(cfg, "eikonal_all_points", False)) and exact.fp_weight is not None:
                sloped = dataclasses.replace(exact, fp_weight=None)   # the slope term counts every sample
            constraints: Dict[str, Tensor] = {
                "eikonal": L.eikonal(sloped),
                "consistency": L.consistency(exact),
                "barrier": L.barrier(fb, cfg),
                "origin": L.origin_anchor(fb, cfg),
                "volume": L.min_volume(fb, cfg),
            }
            pose_terms = L.pose_terms(fb, cfg)
            if batch.anchor_index is not None and bool(getattr(cfg, "anchor_exempt_size", False)):
                # Anchor rows are left out of the size term only: an anchor that was scaled down to fit
                # inside the outer radius is smaller than the size target on purpose, and the anchor
                # pull already fixes its size.
                main_rows = _first_rows(fb, batch.codes.shape[0] - int(batch.anchor_index.numel()))
                pose_terms["pose_scale"] = L.pose_terms(main_rows, cfg)["pose_scale"]
            constraints.update(pose_terms)
            grad_change = L.grad_change(model, fb, cfg, generator=self.generator)
            if graph is not None:
                constraints.update(action_losses(model, fb.mod, fb, graph, cfg))
            else:
                zero = fb.f.new_zeros(())
                constraints.update({key: zero for key in ACTION_KEYS})
            objective, spread_value, spread_corr = self._objective(fb, diversity)
            if batch.pair_first is not None and batch.frame_used >= 0:
                # A frame of partners around code 0: the log-volume of their field changes. Scaled
                # so that weak_frame_weight is its weight in the update, however many passes carry one.
                value, frame_info = L.frame_volume(fb, 0, batch.pair_second, batch.pair_chord,
                                                   batch.frame_used, cfg)
                frames = self._frames()
                objective = objective + float(cfg.weak_frame_weight) * value * parts / frames
                sums["weak/frame_term"] = sums.get("weak/frame_term", 0.0) + float(value.detach()) / frames
                for key, item in frame_info.items():
                    sums[f"weak/{key}"] = sums.get(f"weak/{key}", 0.0) + float(item) / frames
            elif batch.pair_first is not None:
                # Pairs that differ only along little-used directions of code space: a pair
                # counts like one code of the neighbour term.
                pair_sum, pair_info = L.pair_push(fb, batch.pair_first, batch.pair_second,
                                                  batch.pair_chord, cfg)
                pair_count = int(batch.pair_second.numel())
                objective = objective + float(cfg.weak_pair_weight) * pair_sum / (batch.codes.shape[0] - pair_count)
                add("weak/pair_term", pair_sum.detach() / pair_count)
                add("weak/pair_rate", pair_info["pair_rate"])
                add("weak/pair_rate_min", pair_info["pair_rate_min"])
            objective = self._variety_weight_now() * objective
            applied = constraints
            if bool(getattr(cfg, "field_only", False)):
                applied = {key: constraints[key] for key in ("eikonal", "consistency")}
            loss = self.multipliers.combine(objective, applied)
            loss = loss + float(cfg.grad_change_weight) * grad_change
            even = float(getattr(cfg, "code_isotropy_weight", 0.0))
            if even > 0:
                # Every code direction should reach the main network with the same strength.
                uneven, ratio = L.code_isotropy(model, batch.codes, int(cfg.code_isotropy_directions), self.generator)
                loss = loss + even * uneven
                add("iso/term", uneven.detach())
                add("iso/ratio", ratio)
            sharp = float(getattr(cfg, "crease_weight", 0.0))
            if sharp > 0:
                # Flat faces and sharp edges are cheaper than gentle bends (anchor rows left out).
                own = batch.codes.shape[0]
                if batch.anchor_index is not None and not bool(getattr(cfg, "crease_on_anchors", False)):
                    own -= int(batch.anchor_index.numel())
                bend, bend_info = L.crease(model, fb, cfg, generator=self.generator, rows=own)
                loss = loss + sharp * bend
                add("crease/term", bend.detach())
                for key, item in bend_info.items():
                    add(f"crease/{key}", item)
            hold = float(getattr(cfg, "shape_hold_weight", 0.0))
            if hold > 0:
                # Tidy-up stage: the field near the surfaces of a frozen copy of the decoder (taken
                # when the stage began) must not change, so the shapes stay put.
                if getattr(self, "_reference", None) is None:
                    self._reference = copy.deepcopy(model).eval()
                    for p in self._reference.parameters():
                        p.requires_grad_(False)
                newton = int(getattr(cfg, "shape_hold_newton", 0))
                if newton > 0:
                    # Pin the surface itself: slide the sample points onto the copy's surface, then
                    # ask the decoder to stay at zero there. Points that did not arrive are left out.
                    codes = batch.codes.detach()
                    at = fb.x.detach()
                    for _ in range(newton):
                        at = at.requires_grad_(True)
                        value = self._reference(codes, at)
                        (slope,) = torch.autograd.grad(value.sum(), at)
                        size = slope.square().sum(-1, keepdim=True).clamp_min(1e-6)
                        at = (at - value.detach().unsqueeze(-1) * slope / size).detach()
                    with torch.no_grad():
                        left = self._reference(codes, at).abs()
                        inside = at.norm(dim=-1) < float(cfg.bound_radius)
                        arrived = ((left < float(cfg.shape_hold_converged)) & inside).to(fb.f.dtype)
                    moved = (model(codes, at).abs() * arrived).sum() / arrived.sum().clamp_min(1.0)
                    add("hold/arrived", arrived.mean())
                else:
                    with torch.no_grad():
                        before = self._reference(batch.codes.detach(), fb.x)
                        band = (before.abs() < float(cfg.shape_hold_band)).to(fb.f.dtype)
                    moved = ((fb.f - before).abs() * band).sum() / band.sum().clamp_min(1.0)
                loss = loss + hold * moved
                add("hold/moved", moved.detach())
            if batch.anchor_index is not None:
                # Anchor objects: the field at their codes is pulled onto the object's field.
                rows = torch.arange(batch.codes.shape[0] - batch.anchor_index.numel(), batch.codes.shape[0],
                                    device=self.device)
                value, anchor_info = L.anchor_match(fb, rows, self.anchor_grid[batch.anchor_index].float(), cfg,
                                                    weights=batch.anchor_pull)
                loss = loss + self._anchor_weight(step) * value
                add("anchor/term", value.detach())
                add("anchor/weight", self._anchor_weight(step))
                for key, item in anchor_info.items():
                    add(f"anchor/{key}", item)
            if self._budget_on():
                deviation, negative, base_info = self._base_terms(fb)
                loss = (loss + self.multipliers.multiplier("base_deviation") * deviation
                        + float(cfg.base_negative_weight) * negative)
                for key, item in base_info.items():
                    add(key, item)
            if self._spread_on():
                shortfall, spread_info = self._base_spread(fb)
                loss = loss + self.multipliers.multiplier("base_spread") * shortfall
                for key, item in spread_info.items():
                    add(key, item)
            if self._repel_on():
                rows = self._main_rows if self._main_rows is not None else batch.codes.shape[0]
                vec = self._base_vector(fb.mod)[:rows]
                others = getattr(self, "_shared_base", None)
                if others is None:
                    others = vec.new_zeros((0, vec.shape[1]))
                repel, repel_info = L.base_repel(vec, others, cfg, norm=getattr(self, "_main_norm", None))
                loss = loss + float(cfg.base_repel_weight) * repel
                for key, item in repel_info.items():
                    add(key, item)
            if float(getattr(cfg, "correction_cost_weight", 0.0)) > 0 and fb.mod.base is not None:
                own = batch.codes.shape[0] - (0 if batch.anchor_index is None else int(batch.anchor_index.numel()))
                cost, corr_info = self._correction_cost(fb, own)
                loss = loss + float(cfg.correction_cost_weight) * cost
                for key, item in corr_info.items():
                    add(key, item)
            add("spread", spread_value)
            add("spread_corr", spread_corr)
            for key, value in self._stat_spreads.items():
                add(f"stat/{key}", value)
            te = self._sync()
            if bool(torch.isfinite(loss)):
                for key, item in self._backward(loss, fb, constraints, parts).items():
                    add(key, item)
            else:
                finite = False
            tf = self._sync()
            add("loss", loss.detach())
            add("diversity", diversity.detach())
            add("grad_change", grad_change.detach())
            add("e", e)
            add("eps_mean", batch.eps.mean())
            add("graph_on", float(graph is not None))
            for key in CONSTRAINT_KEYS:
                add(f"term/{key}", constraints[key].detach())
            for key in ("mean_ratio", "dead_zone_fraction", "mean_shape_distance",
                        "mean_latent_distance", "phase2", "order_violations", "nearest_distance"):
                add(f"div/{key}", info[key])
            for key, value in _findings(graph).items():
                add(key, value)
            add("time_ms/field", (tb - ta) * 1e3 * parts)
            add("time_ms/graph", (tc - tb) * 1e3 * parts)
            add("time_ms/confirm", (td - tc) * 1e3 * parts)
            add("time_ms/losses", (te - td) * 1e3 * parts)
            add("time_ms/backward", (tf - te) * 1e3 * parts)
            del fb, graph, loss, constraints

        grad_norm = float("nan")
        if finite:
            max_norm = float(cfg.grad_clip) if float(cfg.grad_clip) > 0 else float("inf")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
            if math.isfinite(grad_norm):
                self.optimizer.step()
                self._update_average()
                anchor_lr = self._anchor_lr(step)
                if anchor_lr > 0 and self.anchor_codes.grad is not None:
                    # the anchor codes move on their own, slower, schedule and stay unit vectors
                    torch.nn.utils.clip_grad_norm_([self.anchor_codes], max_norm)
                    for group in self.anchor_optimizer.param_groups:
                        group["lr"] = anchor_lr
                    self.anchor_optimizer.step()
                    with torch.no_grad():
                        self.anchor_codes.copy_(torch.nn.functional.normalize(self.anchor_codes, dim=-1))
            else:
                finite = False
        lr = float(self.optimizer.param_groups[0]["lr"])
        self.scheduler.step()
        # Action terms are only measured when the graph was on for every sub-batch.
        graph_on = sums.pop("graph_on")
        measured = CONSTRAINT_KEYS if graph_on >= 1.0 else FIELD_KEYS
        self.multipliers.update({k: sums[f"term/{k}"] for k in measured})
        used_weight, safe = self._variety_weight_now(), None
        if float(getattr(cfg, "variety_weight_max", 0.0)) > 0:
            safe = self._adapt_variety_weight(bool(finite) and graph_on >= 1.0)
        if self._budget_on() and "base/deviation" in sums:
            self.multipliers.update({"base_deviation": sums["base/deviation"]})
        if self._spread_on() and "base/spread_shortfall" in sums:
            self.multipliers.update({"base_spread": sums["base/spread_shortfall"]})
        t_end = self._sync()

        record: Dict[str, Any] = {
            "step": step,
            "graph_mode": max(set(modes), key=modes.count),
            "graph_built": int(graph_on > 0),
            "skipped": int(not finite),
            "lr": lr,
            "grad_norm": grad_norm,
        }
        record.update(sums)
        if safe is not None:
            record["variety/weight"], record["variety/safe"] = used_weight, float(safe)
        if self._needs_usage():
            record.update(self._usage_info)
        if self.anchor_codes is not None:
            record["anchor/lr"] = self._anchor_lr(step)
        record["tol/consistency"] = self.multipliers.tolerance("consistency")
        for key in CONSTRAINT_KEYS:
            record[f"mult/{key}"] = self.multipliers.multiplier(key)
        if self._budget_on():
            record["mult/base_deviation"] = self.multipliers.multiplier("base_deviation")
            record["tol/base_deviation"] = self.multipliers.tolerance("base_deviation")
        if self._spread_on():
            record["mult/base_spread"] = self.multipliers.multiplier("base_spread")
        record["time_ms/total"] = (t_end - t0) * 1e3
        record["gpu_peak_mb"] = (
            torch.cuda.max_memory_allocated(self.device) / 2 ** 20 if self.device.type == "cuda" else 0.0
        )
        self.batch = None
        self._shared_neighbours = None
        self._shared_base = None
        self._main_rows = None
        self._main_norm = None
        self.step_count += 1
        return record

    def step(self) -> Dict[str, Any]:
        """One optimiser step. Returns the metrics of the step (plain floats and strings)."""
        cfg, model = self.cfg, self.model
        if int(getattr(cfg, "accumulate", 1)) > 1:
            return self._accumulated_step()
        if self.batch is None or self.batch.inner >= max(int(cfg.graph_reuse_steps), 1):
            self.batch = self.sample_batch()
        batch = self.batch
        step = self.step_count
        self._apply_schedules(step)
        model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        t0 = self._sync()
        fb = evaluate_field(model, batch.codes, batch.cluster, batch.x, cfg)
        t_field = self._sync()

        # ---- graph: built on the first step of a batch, then reused
        e = float(L.consistency_error(fb))
        built = False
        t_graph = t_confirm = t_field
        if batch.inner == 0:
            batch.mode = graph_mode_for(e, cfg)
            if batch.mode != "off":
                graph = build_graph(fb, e, cfg, mode=batch.mode)
                t_graph = self._sync()
                batch.graph = confirm(model, fb.mod, graph, cfg)
                self._store_hard_negatives(batch, batch.graph, fb)
                t_confirm = self._sync()
                built = True
        graph = batch.graph

        # ---- terms
        diversity, info = self._diversity(fb, step)
        constraints: Dict[str, Tensor] = {
            "eikonal": L.eikonal(fb),
            "consistency": L.consistency(fb),
            "barrier": L.barrier(fb, cfg),
            "origin": L.origin_anchor(fb, cfg),
            "volume": L.min_volume(fb, cfg),
        }
        # Unbiased split-sample pose terms: noisy, and slightly negative at times.
        constraints.update(L.pose_terms(fb, cfg))
        grad_change = L.grad_change(model, fb, cfg, generator=self.generator)
        if graph is not None:
            constraints.update(action_losses(model, fb.mod, fb, graph, cfg))
        else:
            zero = fb.f.new_zeros(())
            constraints.update({key: zero for key in ACTION_KEYS})

        # Action terms are only measured while the graph is on; their
        # multipliers are left alone otherwise.
        measured = CONSTRAINT_KEYS if graph is not None else FIELD_KEYS
        self.multipliers.update({k: float(constraints[k].detach()) for k in measured})
        objective, spread_value, spread_corr = self._objective(fb, diversity)
        loss = self.multipliers.combine(objective, constraints)
        loss = loss + float(cfg.grad_change_weight) * grad_change
        base_info: Dict[str, float] = {}
        if self._budget_on():
            deviation, negative, base_info = self._base_terms(fb)
            loss = (loss + self.multipliers.multiplier("base_deviation") * deviation
                    + float(cfg.base_negative_weight) * negative)
            self.multipliers.update({"base_deviation": base_info["base/deviation"]})
            base_info["mult/base_deviation"] = self.multipliers.multiplier("base_deviation")
            base_info["tol/base_deviation"] = self.multipliers.tolerance("base_deviation")
        if self._spread_on():
            shortfall, spread_info = self._base_spread(fb)
            loss = loss + self.multipliers.multiplier("base_spread") * shortfall
            self.multipliers.update({"base_spread": spread_info["base/spread_shortfall"]})
            base_info.update(spread_info)
            base_info["mult/base_spread"] = self.multipliers.multiplier("base_spread")
        if self._repel_on():
            vec = self._base_vector(fb.mod)
            repel, repel_info = L.base_repel(vec, vec.new_zeros((0, vec.shape[1])), cfg)
            loss = loss + float(cfg.base_repel_weight) * repel
            base_info.update({k: float(v) for k, v in repel_info.items()})
        if float(getattr(cfg, "correction_cost_weight", 0.0)) > 0 and getattr(fb.mod, "base", None) is not None:
            cost, corr_info = self._correction_cost(fb)
            loss = loss + float(cfg.correction_cost_weight) * cost
            base_info.update(corr_info)
        t_losses = self._sync()

        # ---- optimiser
        self.optimizer.zero_grad(set_to_none=True)
        finite = bool(torch.isfinite(loss))
        grad_norm = float("nan")
        if finite:
            base_info.update(self._backward(loss, fb, constraints, 1.0))
            max_norm = float(cfg.grad_clip) if float(cfg.grad_clip) > 0 else float("inf")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
            if math.isfinite(grad_norm):
                self.optimizer.step()
                self._update_average()
            else:
                finite = False
        lr = float(self.optimizer.param_groups[0]["lr"])
        self.scheduler.step()
        t_end = self._sync()

        # ---- metrics
        record: Dict[str, Any] = {
            "step": step,
            "loss": float(loss.detach()),
            "diversity": float(diversity.detach()),
            "spread": spread_value,
            "spread_corr": spread_corr,
            "grad_change": float(grad_change.detach()),
            "e": e,
            "graph_mode": batch.mode,
            "graph_built": int(built),
            "skipped": int(not finite),
            "lr": lr,
            "grad_norm": grad_norm,
            "eps_mean": float(batch.eps.mean()),
        }
        for key in CONSTRAINT_KEYS:
            record[f"term/{key}"] = float(constraints[key].detach())
            record[f"mult/{key}"] = self.multipliers.multiplier(key)
        for key in ("mean_ratio", "dead_zone_fraction", "mean_shape_distance",
                    "mean_latent_distance", "phase2", "order_violations", "nearest_distance"):
            record[f"div/{key}"] = float(info[key])
        record.update(_findings(graph))
        for key, value in self._stat_spreads.items():
            record[f"stat/{key}"] = value
        record["time_ms/field"] = (t_field - t0) * 1e3
        record["time_ms/graph"] = (t_graph - t_field) * 1e3
        record["time_ms/confirm"] = (t_confirm - t_graph) * 1e3
        record["time_ms/losses"] = (t_losses - t_confirm) * 1e3
        record["time_ms/backward"] = (t_end - t_losses) * 1e3
        record["time_ms/total"] = (t_end - t0) * 1e3
        record["gpu_peak_mb"] = (
            torch.cuda.max_memory_allocated(self.device) / 2 ** 20 if self.device.type == "cuda" else 0.0
        )
        record.update(base_info)

        batch.inner += 1
        self.step_count += 1
        return record

    # ----------------------------------------------------------- diagnostics
    def probe_codes(self) -> Tensor:
        return self.bank.codes[: max(int(self.cfg.probe_codes), 1)]

    def run_diagnostics(self) -> Dict[str, float]:
        """Grid topology, graph-vs-grid, dead directions and descriptor spread on the probes.

        Uses its own generator (same points every time), so it never disturbs
        the training random stream.
        """
        from . import diagnostics as D

        cfg, model = self.cfg, self.model
        was_training = model.training
        model.eval()
        t0 = self._sync()
        gen = torch.Generator().manual_seed(int(cfg.seed) + 7919)
        codes = self.probe_codes()
        out: Dict[str, float] = {"step": self.step_count}

        # field statistics of the probes at shared points
        x = sample_points(cfg, 1, gen, self.device)
        cluster = torch.zeros(codes.shape[0], dtype=torch.long, device=self.device)
        fb = evaluate_field(model, codes, cluster, x, cfg, create_graph=False)
        with torch.no_grad():
            e = float(L.consistency_error(fb))
            out["e"] = e
            out["eikonal"] = float(L.eikonal(fb))
            out["consistency"] = float(L.consistency(fb))
            stats = L.descriptors(fb, cfg)
            scales = torch.as_tensor(L.DESCRIPTOR_SCALES, dtype=stats.dtype, device=stats.device)
            spread = stats.std(dim=0) if stats.shape[0] > 1 else torch.zeros_like(scales)
            for name, mean, std in zip(L.DESCRIPTOR_NAMES, stats.mean(dim=0), spread):
                out[f"descriptor_mean/{name}"] = float(mean)
                out[f"descriptor_std/{name}"] = float(std)
            out["descriptor_spread"] = float((spread / scales).mean())
            # global diversity: mean |f_i - f_j| between probe codes inside R_b
            inside_bound = (fb.x[0].square().sum(dim=-1) < float(cfg.bound_radius) ** 2)
            values = fb.f[:, inside_bound].detach()
            if values.shape[0] > 1:
                pair = torch.cdist(values, values, p=1) / max(values.shape[1], 1)
                count = values.shape[0]
                out["probe_field_distance"] = float(pair.sum() / (count * (count - 1)))
            else:
                out["probe_field_distance"] = 0.0
        del fb

        if getattr(model, "box_base", False):
            out.update(self._probe_base(codes, x, gen))

        # grid topology and the graph-versus-grid check
        mode = "sign" if e > float(cfg.graph_full_error) else "full"
        check = D.graph_vs_grid(model, codes, cfg, resolution=int(cfg.diag_resolution),
                                generator=gen, mode=mode)
        grid = check["grid_counts"]
        count = max(len(grid), 1)
        for key in ("pieces", "cavities", "handles"):
            out[f"grid/{key}_mean"] = sum(r[key] for r in grid) / count
        out["grid/one_piece_fraction"] = sum(r["pieces"] == 1 for r in grid) / count
        out["grid/no_cavity_fraction"] = sum(r["cavities"] == 0 for r in grid) / count
        out["grid/no_handle_fraction"] = sum(r["handles"] == 0 for r in grid) / count
        out["grid/clean_fraction"] = sum(
            r["pieces"] == 1 and r["cavities"] == 0 and r["handles"] == 0 for r in grid) / count
        out["grid/volume_fraction_mean"] = sum(r["volume_fraction"] for r in grid) / count
        out["grid/volume_fraction_std"] = float(
            torch.tensor([r["volume_fraction"] for r in grid]).std()) if len(grid) > 1 else 0.0
        for key in ("pieces_agreement", "cavities_agreement", "both_agreement"):
            out[f"graph_vs_grid/{key}"] = check[key]
        for key, dropped in check["confirm_dropped"].items():
            out[f"graph_vs_grid/confirm_dropped_{key}"] = dropped
        out["graph_vs_grid/raw_extra_pieces"] = int(
            (check["graph_counts_raw"]["pieces"] - 1).clamp_min(0).sum())
        out["graph_vs_grid/raw_cavities"] = int(check["graph_counts_raw"]["cavities"].sum())

        # pose of the probes from a large sample (the training values are noisy)
        out.update(self._probe_pose(codes, gen))

        # dead directions on a few probes
        results, stability, live_pair = [], [], []
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

        if self.anchor_codes is not None:
            out.update(self._probe_anchors(gen))
        out["time_s"] = self._sync() - t0
        model.train(was_training)
        return out

    @torch.no_grad()
    def _probe_base(self, codes: Tensor, x: Tensor, generator: torch.Generator, fresh: int = 1024) -> Dict[str, float]:
        """Rounded-box base: its parameters over fresh random codes, and its share of the variety.

        ``base/field_variance_share`` is the variance across the probe codes of the base field
        alone, over that of the whole field (mean over the shared points inside the bounding
        radius). Near 1, the variety lives in the base; the residual adds little of its own.
        """
        model = self.model
        z = sample_sphere(fresh, int(self.cfg.latent_dim), generator, self.device)
        base = model.prepare(z).base
        b, r = base[:, :3], base[:, 3]
        reach = b[:, 0] + r
        thin = b[:, 2] + r
        out = {"base/range": float(model.base_range)}
        for name, value in (("r", r), ("b1", b[:, 0]), ("b2", b[:, 1]), ("b3", b[:, 2]), ("reach", reach),
                            ("thickness", thin), ("elongation", reach / thin)):
            out[f"base/{name}_mean"] = float(value.mean())
            out[f"base/{name}_min"] = float(value.min())
            out[f"base/{name}_max"] = float(value.max())
        out["base/rod_fraction"] = float(((b[:, 0] > 2.0 * r) & (b[:, 1] < 0.5 * r)).float().mean())
        out["base/sheet_fraction"] = float(((b[:, 1] > 2.0 * r) & (b[:, 2] < 0.5 * r)).float().mean())
        out["base/thin_fraction"] = float((thin < 0.15).float().mean())
        deform = model.prepare(z).deform
        if deform is not None:
            bend, taper, shift = deform[:, :6], deform[:, 6:9], deform[:, 9:12]
            out["base/bend_abs_mean"] = float(bend.abs().mean())
            out["base/bend_abs_max"] = float(bend.abs().max())
            out["base/bend_std"] = float(bend.std(dim=0).mean())
            out["base/taper_abs_mean"] = float(taper.abs().mean())
            out["base/taper_abs_max"] = float(taper.abs().max())
            out["base/shift_mean"] = float(shift.norm(dim=-1).mean())
            # how well the centring works: soft centroid of the final base on a fine grid, first 16 codes
            sub = model.prepare(z[:16])
            axis = torch.linspace(-0.9, 0.9, 48, device=z.device)
            grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3).expand(16, -1, -1)
            w = torch.sigmoid(model.base_field(sub, grid) / float(self.cfg.base_center_tau))
            residual = (w[..., None] * grid).sum(1) / w.sum(1).clamp_min(1e-6)[:, None]
            out["base/centroid_residual_mean"] = float(residual.norm(dim=-1).mean())
        mod = model.prepare(codes)
        points = x.expand(codes.shape[0], -1, -1)
        inside = points[0].square().sum(-1) < float(self.cfg.bound_radius) ** 2
        whole = model.query(mod, points)[:, inside]
        alone = model.base_field(mod, points)[:, inside]
        if codes.shape[0] > 1:
            total = float(whole.var(dim=0).mean())
            out["base/field_variance_share"] = float(alone.var(dim=0).mean()) / max(total, 1e-12)
        return out

    @torch.no_grad()
    def _probe_anchors(self, generator: torch.Generator, points: int = 8192) -> Dict[str, float]:
        """How well every anchor object is reproduced at its code, and how far apart the codes are."""
        cfg = self.cfg
        x = (torch.rand(points, 3, generator=generator) * 2.0 - 1.0).to(self.device)
        inside = x.square().sum(dim=-1) < float(cfg.bound_radius) ** 2
        codes = torch.nn.functional.normalize(self.anchor_codes.detach(), dim=-1)
        distance, overlap = [], []
        for s in range(0, codes.shape[0], 16):
            block = codes[s:s + 16]
            where = x.expand(block.shape[0], -1, -1)
            f = self.model(block, where)
            target = L.sample_grid(self.anchor_grid[s:s + 16].float(), where)
            distance.append(((f - target).abs() * inside).sum(dim=1) / inside.sum().clamp_min(1))
            ours, theirs = f > 0, target > 0
            overlap.append((ours & theirs).sum(dim=1) / (ours | theirs).sum(dim=1).clamp_min(1))
        distance, overlap = torch.cat(distance), torch.cat(overlap)
        cosine = codes @ codes.T
        cosine = cosine[~torch.eye(codes.shape[0], dtype=torch.bool, device=codes.device)]
        return {"anchor/distance_mean": float(distance.mean()), "anchor/distance_max": float(distance.max()),
                "anchor/iou_mean": float(overlap.mean()), "anchor/iou_min": float(overlap.min()),
                "anchor/code_cos_abs_mean": float(cosine.abs().mean()) if cosine.numel() else 0.0,
                "anchor/code_cos_max": float(cosine.max()) if cosine.numel() else 0.0}

    @torch.no_grad()
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
        # the same shapes under the rule in force (cfg.pose_power): offset of the centre and size as a length
        power = float(getattr(cfg, "pose_power", 2.0))
        centre, moment = L.extent_pose(L.soft_inside(fake, cfg), fake.x, power)
        size, size_target = moment.pow(1.0 / power), L.pose_size_target(cfg)
        return {
            "pose/power": power,
            "pose/extent_centre_offset_mean": float(centre.norm(dim=-1).mean()),
            "pose/extent_centre_offset_max": float(centre.norm(dim=-1).max()),
            "pose/extent_size_mean": float(size.mean()),
            "pose/extent_size_target": float(size_target),
            "pose/extent_size_rel_error_rms": float(((size - size_target) / size_target).square().mean().sqrt()),
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

    # ------------------------------------------------- optimiser groups and weight average
    def _group_members(self) -> Dict[str, set]:
        """Parameter ids of the special groups: "base" (the learned base head) and "out" (final layer)."""
        model = self.model
        members: Dict[str, set] = {}
        for name in ("base_head", "base_head_mlp", "base_net"):
            if hasattr(model, name):
                members["base"] = {id(p) for p in getattr(model, name).parameters()}
        members["out"] = {id(p) for p in model.out.parameters()}
        return members

    def _group_names(self) -> List[str]:
        """Names of the optimiser's groups, in order: "main", then "base" and "out" when they have a rate."""
        cfg = self.cfg
        names = ["main"]
        if float(getattr(cfg, "base_lr_scale", 1.0)) != 1.0 and "base" in self._group_members():
            names.append("base")
        if float(getattr(cfg, "out_lr_scale", 1.0)) != 1.0:
            names.append("out")
        return names

    def _split(self, names: Sequence[str]) -> Dict[str, List[Tensor]]:
        """The model's parameters (in model order) split into the groups ``names`` ("main" takes the rest)."""
        members = self._group_members()
        out: Dict[str, List[Tensor]] = {name: [] for name in names}
        for p in self.model.parameters():
            home = next((n for n in names if n != "main" and id(p) in members.get(n, set())), "main")
            out[home].append(p)
        return out

    def _param_groups(self) -> List[Dict[str, Any]]:
        """One group per name in ``_group_names``: the rest, the learned base head (``base_lr_scale``) and the
        final layer (``out_lr_scale``), each with its own fraction of the learning rate."""
        cfg = self.cfg
        scales = {"main": 1.0, "base": float(getattr(cfg, "base_lr_scale", 1.0)),
                  "out": float(getattr(cfg, "out_lr_scale", 1.0))}
        names = self._group_names()
        split = self._split(names)
        groups = []
        for name in names:
            group = {"params": split[name], "lr_scale": scales[name], "name": name}
            if scales[name] != 1.0:
                group["lr"] = float(cfg.lr) * scales[name]
            groups.append(group)
        return groups

    def _optimizer_order(self, saved_groups: Sequence[dict]) -> List[Tensor]:
        """The model's parameters in the order a saved optimiser numbered them.

        Groups saved before they had names: one group is everything; two are the rest and the final layer.
        """
        names = [g.get("name") for g in saved_groups]
        if any(n is None for n in names):
            names = ["main"] if len(saved_groups) == 1 else ["main", "out"]
        split = self._split(names)
        return [p for name in names for p in split[name]]

    def _load_optimizer(self, saved: dict) -> None:
        """Load the optimiser state, also when the saved run had other groups (carried over per parameter)."""
        saved_names = [g.get("name") for g in saved["param_groups"]]
        if len(saved["param_groups"]) == len(self.optimizer.param_groups) and (
                saved_names == [g.get("name") for g in self.optimizer.param_groups]
                or (all(n is None for n in saved_names) and self._group_names() in (["main"], ["main", "out"]))):
            self.optimizer.load_state_dict(saved)
            for group, name in zip(self.optimizer.param_groups, self._group_names()):
                group["name"] = name
            self._drop_mismatched_state()
            return
        order = self._optimizer_order(saved["param_groups"])
        self.optimizer.state.clear()
        for index, value in saved["state"].items():
            p = order[int(index)]
            self.optimizer.state[p] = {key: (item.to(p.device) if torch.is_tensor(item) and item.dim() > 0 else item)
                                       for key, item in value.items()}
        self._drop_mismatched_state()

    def _drop_mismatched_state(self) -> None:
        """Adam state of a parameter that has grown since it was saved starts fresh."""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                st = self.optimizer.state.get(p)
                if st and torch.is_tensor(st.get("exp_avg")) and st["exp_avg"].shape != p.shape:
                    del self.optimizer.state[p]

    def _copy_for_average(self) -> torch.nn.Module:
        average = copy.deepcopy(self.model).eval()
        for p in average.parameters():
            p.requires_grad_(False)
        return average

    @torch.no_grad()
    def _update_average(self) -> None:
        if self.average is None:
            return
        keep = float(self.cfg.weight_average_decay)
        for a, p in zip(self.average.parameters(), self.model.parameters()):
            a.lerp_(p.detach(), 1.0 - keep)

    # ----------------------------------------------------------- checkpoints
    def state_dict(self) -> dict:
        state = {
            "config": self.cfg.to_dict(),
            "step": self.step_count,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "multipliers": self.multipliers.state_dict(),
            "variety_weight": getattr(self, "_variety_weight", None),
            "bank": self.bank.state_dict(),
            "generator": self.generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "batch": self.batch,
            "usage": self._usage,
        }
        if self.anchor_codes is not None:
            state["anchor_codes"] = self.anchor_codes.detach().clone()
            state["anchor_optimizer"] = self.anchor_optimizer.state_dict()
        if self.device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state(self.device)
        if self.average is not None:
            state["average"] = self.average.state_dict()
        return state

    def _copy_grown(self, weights: dict) -> List[str]:
        """Load saved weights into the model, letting a tensor have grown since: extra input columns
        (dim 1) or extra output rows (dim 0) keep their start (zero for the base head's last layer and
        the base-feature columns). Returns the names that grew."""
        own = self.model.state_dict()
        unknown = [key for key in weights if key not in own]
        missing = [key for key in own if key not in weights and key != "base_range"]
        if unknown or missing:
            raise ValueError(f"checkpoint does not match the model: unknown {unknown[:6]}, missing {missing[:6]}")
        grown = []
        with torch.no_grad():
            for key, value in weights.items():
                target = own[key]
                value = value.to(device=target.device, dtype=target.dtype)
                if value.shape == target.shape:
                    target.copy_(value)
                elif value.dim() == target.dim() and all(a <= b for a, b in zip(value.shape, target.shape)):
                    target[tuple(slice(0, n) for n in value.shape)].copy_(value)
                    grown.append(key)
                else:
                    raise ValueError(f"{key}: saved {tuple(value.shape)}, model {tuple(target.shape)}")
        return grown

    def load_state_dict(self, state: dict) -> None:
        self.step_count = int(state["step"])
        grown = self._copy_grown(state["model"])
        std = float(getattr(self.cfg, "base_deform_init_std", 0.0))
        if std > 0 and getattr(self.model, "base_deform", False):
            for name in ("base_head", "base_head_mlp", "base_net"):
                if hasattr(self.model, name):
                    last = getattr(self.model, name)
                    last = last[-1] if isinstance(last, torch.nn.Sequential) else last
                    key = [k for k, p in self.model.named_parameters() if p is last.weight][0]
                    if key in grown:
                        with torch.no_grad():
                            gen = torch.Generator().manual_seed(int(self.cfg.seed) + 4242)
                            rows = last.weight[4:]
                            rows.copy_(torch.randn(rows.shape, generator=gen).to(rows) * std)
        self._load_optimizer(state["optimizer"])
        scheduler = dict(state["scheduler"])
        scheduler["lr_lambdas"] = [None]  # the schedule follows the current config
        self.scheduler.load_state_dict(scheduler)
        # The base learning rate follows the current config as well, so `--set lr=...` on a resume
        # takes effect (the saved optimiser and scheduler state would otherwise keep the old one).
        # Each group keeps its own fraction of it (the final layer's out_lr_scale).
        scales = [float(group.get("lr_scale", 1.0)) for group in self._param_groups()]
        self.scheduler.base_lrs = [float(self.cfg.lr) * s for s in scales]
        for group, s in zip(self.optimizer.param_groups, scales):
            group["lr_scale"] = s
            group["initial_lr"] = float(self.cfg.lr) * s
            group["lr"] = float(self.cfg.lr) * s * lr_factor(self.step_count, self.cfg)
        if self.average is not None:
            if state.get("average") is not None:
                self.average.load_state_dict(state["average"])
            else:                                       # averaging starts from the loaded weights
                self.average.load_state_dict(self.model.state_dict())
        self.multipliers.load_state_dict(state["multipliers"])
        self._variety_weight = state.get("variety_weight")
        self.bank.load_state_dict(state["bank"])
        # map_location may have moved these byte tensors; RNG states live on the CPU
        self.generator.set_state(state["generator"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda" and "cuda_rng" in state:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        self.batch = _batch_to(state["batch"], self.device)
        if state.get("usage") is not None:
            self._set_usage(state["usage"])
        if self.anchor_codes is not None and state.get("anchor_codes") is not None:
            if state["anchor_codes"].shape != self.anchor_codes.shape:
                raise ValueError("the checkpoint's anchor codes do not match the anchor objects in use")
            with torch.no_grad():
                self.anchor_codes.copy_(state["anchor_codes"].to(self.device))
            self.anchor_optimizer.load_state_dict(state["anchor_optimizer"])

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Write the full resume state; returns the path."""
        if path is None:
            if self.output_dir is None:
                raise ValueError("no output_dir: give a path")
            path = os.path.join(self.output_dir, CHECKPOINT_NAME)
        state = self.state_dict()
        if self.average is not None:
            # The same resume state with the averaged weights as the model, written first so that a
            # copy taken when checkpoint_last.pt changes is never older than it.
            averaged = dict(state, model=self.average.state_dict(), averaged_weights=True)
            other = os.path.join(os.path.dirname(path), "checkpoint_average_last.pt")
            torch.save(averaged, other + ".tmp")
            os.replace(other + ".tmp", other)
        temporary = path + ".tmp"
        torch.save(state, temporary)
        os.replace(temporary, path)
        return path

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: typing.Union[str, torch.device] = "cpu",
        output_dir: Optional[str] = None,
        overrides: Optional[Sequence[str]] = None,
        tensorboard: bool = True,
    ) -> "Trainer":
        """Trainer restored from a checkpoint; ``overrides`` change its config first."""
        state = torch.load(path, map_location=device, weights_only=False)
        cfg = apply_overrides(DataFreeConfig.from_dict(state["config"]), overrides)
        # the starting weights are not needed again (the checkpoint has the model), so the file
        # they came from does not have to exist any more
        start, cfg.init_main_from = getattr(cfg, "init_main_from", None), None
        whole, cfg.init_model_from = getattr(cfg, "init_model_from", None), None
        trainer = cls(cfg, device=device, output_dir=output_dir, tensorboard=tensorboard)
        trainer.cfg.init_main_from = start
        trainer.cfg.init_model_from = whole
        trainer.load_state_dict(state)
        return trainer

    # --------------------------------------------------------------- logging
    def _open_logs(self) -> None:
        if self.output_dir is None or self._metrics_file is not None:
            return
        self._metrics_file = open(os.path.join(self.output_dir, "metrics.jsonl"), "a")
        self._diag_file = open(os.path.join(self.output_dir, "diagnostics.jsonl"), "a")
        if self._tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(os.path.join(self.output_dir, "tb"),
                                         purge_step=self.step_count)

    def close(self) -> None:
        for handle in (self._metrics_file, self._diag_file, self._writer):
            if handle is not None:
                handle.close()
        self._metrics_file = self._diag_file = self._writer = None

    def log_step(self, record: Dict[str, Any], scalars: bool) -> None:
        """Append to ``metrics.jsonl``; with ``scalars`` also write TensorBoard scalars."""
        if self._metrics_file is not None:
            self._metrics_file.write(json.dumps(record) + "\n")
        if scalars and self._writer is not None:
            for key, value in record.items():
                if key == "step":
                    continue
                if key == "graph_mode":
                    value = GRAPH_MODES.index(value)
                self._writer.add_scalar(f"train/{key}", value, record["step"])

    def log_diagnostics(self, record: Dict[str, float]) -> None:
        if self._diag_file is not None:
            self._diag_file.write(json.dumps(record) + "\n")
            self._diag_file.flush()
        if self._writer is not None:
            for key, value in record.items():
                if key != "step":
                    self._writer.add_scalar(f"diag/{key}", value, record["step"])

    # ------------------------------------------------------------------- run
    def run(self, steps: Optional[int] = None, verbose: bool = True) -> None:
        """Train up to ``cfg.steps`` optimiser steps in total (or ``steps`` more)."""
        cfg = self.cfg
        last = int(cfg.steps) if steps is None else self.step_count + int(steps)
        self._open_logs()
        diag_done = -1
        try:
            while self.step_count < last:
                if int(cfg.diag_every) > 0 and self.step_count % int(cfg.diag_every) == 0:
                    diag_done = self._diagnose(verbose)
                record = self.step()
                if getattr(self.model, "box_base", False):
                    record["base_range"] = float(self.model.base_range)
                done = self.step_count
                scalars = record["step"] % max(int(cfg.log_every), 1) == 0 or done == last
                self.log_step(record, scalars)
                if scalars and verbose:
                    print(summary_line(record), flush=True)
                if int(cfg.ckpt_every) > 0 and done % int(cfg.ckpt_every) == 0 and done < last:
                    self.save_checkpoint()
            if int(cfg.diag_every) > 0 and diag_done != self.step_count:
                self._diagnose(verbose)
            if self.output_dir is not None:
                self.save_checkpoint()
        finally:
            if self._metrics_file is not None:
                self._metrics_file.flush()
            self.close()

    def _diagnose(self, verbose: bool) -> int:
        record = self.run_diagnostics()
        self.log_diagnostics(record)
        if verbose:
            print(diagnostics_line(record), flush=True)
        return self.step_count


def train_step(trainer: Trainer) -> Dict[str, Any]:
    """One optimiser step of ``trainer`` (function form of :meth:`Trainer.step`)."""
    return trainer.step()


# ----------------------------------------------------------------------- helpers
def _findings(graph: Optional[ShapeGraph]) -> Dict[str, float]:
    """Counts of what the graph in use reports after ``confirm``, over the whole batch."""
    if graph is None:
        return {f"graph/{k}": 0 for k in (
            "fragments_kind0", "fragments_kind1", "cavities", "gap_sites", "wall_sites",
            "extra_pieces", "cavities_uncapped",
            "confirm_dropped_kind0", "confirm_dropped_kind1", "confirm_dropped_cavities",
            "build_ms")}
    stats = graph.stats
    return {
        "graph/fragments_kind0": int((graph.frag_kind == 0).sum()),
        "graph/fragments_kind1": int((graph.frag_kind == 1).sum()),
        "graph/cavities": int(graph.cav_shape.numel()),
        "graph/gap_sites": int(graph.gap_shape.numel()),
        "graph/wall_sites": int(graph.wall_shape.numel()),
        # uncapped counts kept up to date by confirm (pieces beyond the main body)
        "graph/extra_pieces": int((graph.num_pieces - 1).clamp_min(0).sum())
        if graph.num_pieces is not None else 0,
        "graph/cavities_uncapped": int(graph.num_cavities.sum())
        if graph.num_cavities is not None else 0,
        "graph/confirm_dropped_kind0": int(stats.get("confirm_dropped_kind0", 0)),
        "graph/confirm_dropped_kind1": int(stats.get("confirm_dropped_kind1", 0)),
        "graph/confirm_dropped_cavities": int(stats.get("confirm_dropped_cavities", 0)),
        "graph/build_ms": float(stats.get("total_ms", 0.0)),
    }


def _batch_to(batch: Optional[Batch], device) -> Optional[Batch]:
    """A checkpointed batch (and its graph) moved to ``device``."""
    if batch is None:
        return None

    def move(obj):
        changes = {
            f.name: getattr(obj, f.name).to(device)
            for f in dataclasses.fields(obj)
            if isinstance(getattr(obj, f.name), Tensor)
        }
        return dataclasses.replace(obj, **changes)

    moved = move(batch)
    if moved.graph is not None:
        moved.graph = move(moved.graph)
    return moved


def summary_line(record: Dict[str, Any]) -> str:
    return (
        f"step {record['step']:6d}  loss {record['loss']:.4g}  div {record['diversity']:.4g} "
        f"(ratio {record['div/mean_ratio']:.3f}, dead zone {record['div/dead_zone_fraction']:.2f})  "
        f"eik {record['term/eikonal']:.3g}  cons {record['term/consistency']:.3g}  "
        f"pose c {record['term/pose_centroid']:.2g} s {record['term/pose_scale']:.2g} "
        f"a {record['term/pose_axes']:.2g}  e {record['e']:.3g}  graph {record['graph_mode']} "
        f"[frag {record['graph/fragments_kind0']}+{record['graph/fragments_kind1']} "
        f"cav {record['graph/cavities']} gap {record['graph/gap_sites']} "
        f"wall {record['graph/wall_sites']}]  lr {record['lr']:.2e}  "
        f"|g| {record['grad_norm']:.3g}  {record['time_ms/total']:.0f} ms"
        + (f"  anchors d {record['anchor/distance']:.4f} iou {record['anchor/iou']:.2f}"
           if "anchor/distance" in record else "")
    )


def diagnostics_line(record: Dict[str, float]) -> str:
    return (
        f"diag {int(record['step']):6d}  clean {record['grid/clean_fraction']:.2f} "
        f"(1 piece {record['grid/one_piece_fraction']:.2f}, no cavity "
        f"{record['grid/no_cavity_fraction']:.2f}, no handle {record['grid/no_handle_fraction']:.2f})  "
        f"graph=grid {record['graph_vs_grid/both_agreement']:.2f}  "
        f"live {record.get('dead/live_mean', float('nan')):.1f}  "
        f"wobble {record.get('dead/wobble_share_mean', float('nan')):.2f}  "
        f"descriptor spread {record['descriptor_spread']:.3f}  "
        f"probe distance {record['probe_field_distance']:.4f}  {record['time_s']:.1f} s"
        + (f"  anchors iou {record['anchor/iou_mean']:.2f} (min {record['anchor/iou_min']:.2f}) "
           f"code cos max {record['anchor/code_cos_max']:.2f}" if "anchor/iou_mean" in record else "")
        + (f"  base r {record['base/r_mean']:.2f} (min {record['base/r_min']:.2f}) elongation "
           f"{record['base/elongation_mean']:.2f} (max {record['base/elongation_max']:.2f}) variance share "
           f"{record.get('base/field_variance_share', float('nan')):.2f}" if "base/r_mean" in record else "")
    )


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the data-free latent space decoder.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=None,
                        help="total optimiser steps (default: the config's `steps`)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help=f"continue from <output-dir>/{CHECKPOINT_NAME}")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config field (repeatable), e.g. --set lr=1e-4 "
                             "--set tolerances.eikonal=0.02")
    parser.add_argument("--config-from", default=None, metavar="CONFIG_JSON",
                        help="start a new run from another run's config.json (then --set overrides); "
                             "e.g. with --set init_model_from=<its checkpoint> to continue it under new settings")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Trainer:
    args = build_parser().parse_args(argv)
    overrides: List[str] = list(args.overrides)
    if args.steps is not None:
        overrides.append(f"steps={args.steps}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    tensorboard = not args.no_tensorboard
    checkpoint = os.path.join(args.output_dir, CHECKPOINT_NAME)
    if args.resume:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"--resume: no checkpoint at {checkpoint}")
        trainer = Trainer.from_checkpoint(checkpoint, device=args.device, output_dir=args.output_dir,
                                          overrides=overrides, tensorboard=tensorboard)
        if not args.quiet:
            print(f"resumed from {checkpoint} at step {trainer.step_count}", flush=True)
    else:
        start = DataFreeConfig()
        if args.config_from:
            with open(args.config_from) as handle:
                start = DataFreeConfig.from_dict(json.load(handle))
        cfg = apply_overrides(start, overrides)
        trainer = Trainer(cfg, device=args.device, output_dir=args.output_dir, tensorboard=tensorboard)
        with open(os.path.join(args.output_dir, "config.json"), "w") as handle:
            json.dump(cfg.to_dict(), handle, indent=2)
    trainer.run(verbose=not args.quiet)
    return trainer


if __name__ == "__main__":
    main()
