"""Adaptive constraint multipliers for the data-free latent space trainer.

Diversity is the objective; every other term is a constraint with a tolerance.
Each constraint has one multiplier.  It grows geometrically while a running
average of the constraint value is above the tolerance and decays (at half the
rate) once the tolerance is met.  Multipliers are plain Python floats: no
gradient ever flows through them.

Every constraint may have its own range (``mult_ranges[name] = (low, high)``)
and its own rate (``mult_rates[name]``); the global ``mult_min`` / ``mult_max``
/ ``mult_rate`` apply to the names not listed.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Tuple

from torch import Tensor

__all__ = ["AdaptiveMultipliers"]


def _to_float(value) -> float:
    if isinstance(value, Tensor):
        return float(value.detach())
    return float(value)


class AdaptiveMultipliers:
    """One multiplier per named constraint.

    Parameters
    ----------
    cfg:
        A ``DataFreeConfig`` (reads ``tolerances``, ``mult_init``, ``mult_min``,
        ``mult_max``, ``mult_rate``, ``mult_ema``).  May be ``None`` when every
        hyperparameter is given by keyword.  ``mult_ranges`` and ``mult_rates``
        (per-constraint overrides) are read from it too when present.
    names:
        Optional constraint names to create up front.  Names not listed are
        created on first use (in ``update``, ``combine`` or ``multiplier``) at
        ``mult_init``.
    default_tolerance:
        Tolerance used for a constraint that has no entry in ``tolerances``.
    """

    def __init__(
        self,
        cfg=None,
        names: Optional[Iterable[str]] = None,
        *,
        tolerances: Optional[Mapping[str, float]] = None,
        mult_init: Optional[float] = None,
        mult_min: Optional[float] = None,
        mult_max: Optional[float] = None,
        mult_rate: Optional[float] = None,
        mult_ema: Optional[float] = None,
        default_tolerance: float = 0.0,
        mult_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
        mult_rates: Optional[Mapping[str, float]] = None,
    ) -> None:
        def pick(explicit, attr, fallback):
            if explicit is not None:
                return explicit
            if cfg is not None and getattr(cfg, attr, None) is not None:
                return getattr(cfg, attr)
            return fallback

        self.tolerances: Dict[str, float] = {
            str(k): float(v) for k, v in dict(pick(tolerances, "tolerances", {})).items()
        }
        self.mult_init = float(pick(mult_init, "mult_init", 1.0))
        self.mult_min = float(pick(mult_min, "mult_min", 1e-2))
        self.mult_max = float(pick(mult_max, "mult_max", 1e4))
        self.mult_rate = float(pick(mult_rate, "mult_rate", 0.02))
        self.mult_ema = float(pick(mult_ema, "mult_ema", 0.95))
        self.default_tolerance = float(default_tolerance)
        if not self.mult_min <= self.mult_max:
            raise ValueError("mult_min must not exceed mult_max")
        self.mult_ranges: Dict[str, Tuple[float, float]] = {
            str(k): (float(v[0]), float(v[1]))
            for k, v in dict(pick(mult_ranges, "mult_ranges", {})).items()
        }
        for key, (low, high) in self.mult_ranges.items():
            if not low <= high:
                raise ValueError(f"mult_ranges[{key!r}]: low must not exceed high")
        self.mult_rates: Dict[str, float] = {
            str(k): float(v) for k, v in dict(pick(mult_rates, "mult_rates", {})).items()
        }
        self._mult: Dict[str, float] = {}
        self._ema: Dict[str, float] = {}
        for name in names or ():
            self.multiplier(name)

    # ------------------------------------------------------------------ access
    def range(self, name: str) -> Tuple[float, float]:
        """``(low, high)`` of the multiplier of ``name``."""
        return self.mult_ranges.get(name, (self.mult_min, self.mult_max))

    def rate(self, name: str) -> float:
        """Growth rate of the multiplier of ``name`` (it decays at half of it)."""
        return self.mult_rates.get(name, self.mult_rate)

    def _clamp(self, value: float, name: Optional[str] = None) -> float:
        low, high = (self.mult_min, self.mult_max) if name is None else self.range(name)
        return min(max(value, low), high)

    def multiplier(self, name: str) -> float:
        """Current multiplier of ``name`` (created at ``mult_init`` if new)."""
        if name not in self._mult:
            self._mult[name] = self._clamp(self.mult_init, name)
        return self._mult[name]

    def tolerance(self, name: str) -> float:
        return self.tolerances.get(name, self.default_tolerance)

    def ema(self, name: str) -> Optional[float]:
        """Running average of the constraint value; ``None`` before any update."""
        return self._ema.get(name)

    def multipliers(self) -> Dict[str, float]:
        return dict(self._mult)

    def emas(self) -> Dict[str, float]:
        return dict(self._ema)

    def violated(self, name: str) -> bool:
        ema = self._ema.get(name)
        return ema is not None and ema > self.tolerance(name)

    # ------------------------------------------------------------------ update
    def update(self, values: Mapping[str, float]) -> Dict[str, float]:
        """Fold new constraint values into the EMAs and adapt the multipliers.

        ``values`` maps names to floats (0-dim tensors are accepted and
        detached).  A non-finite value is ignored: its EMA and multiplier are
        left unchanged.  The first value of a constraint initialises its EMA.
        Returns the multipliers after the update.
        """
        for name, raw in values.items():
            value = _to_float(raw)
            lam = self.multiplier(name)
            if not math.isfinite(value):
                continue
            previous = self._ema.get(name)
            if previous is None:
                ema = value
            else:
                ema = self.mult_ema * previous + (1.0 - self.mult_ema) * value
            self._ema[name] = ema
            rate = self.rate(name)
            if ema > self.tolerance(name):
                lam *= 1.0 + rate
            else:
                lam *= 1.0 - 0.5 * rate
            self._mult[name] = self._clamp(lam, name)
        return self.multipliers()

    # ----------------------------------------------------------------- combine
    def combine(self, objective: Tensor, constraints: Mapping[str, Tensor]) -> Tensor:
        """``objective + sum_k lambda_k * c_k`` with float multipliers."""
        total = objective
        for name, value in constraints.items():
            total = total + self.multiplier(name) * value
        return total

    # ------------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {
            "multipliers": dict(self._mult),
            "ema": dict(self._ema),
            "tolerances": dict(self.tolerances),
            "default_tolerance": self.default_tolerance,
            "mult_init": self.mult_init,
            "mult_min": self.mult_min,
            "mult_max": self.mult_max,
            "mult_rate": self.mult_rate,
            "mult_ema": self.mult_ema,
            "mult_ranges": {k: tuple(v) for k, v in self.mult_ranges.items()},
            "mult_rates": dict(self.mult_rates),
        }

    def load_state_dict(self, state: Mapping, load_hyperparameters: bool = False) -> None:
        """Restore multipliers and EMAs.

        Tolerances, ranges and rates stay those of the current config unless
        ``load_hyperparameters`` is set, so a resumed run can change them.
        Multipliers are clamped to the ranges in force after loading.
        """
        if load_hyperparameters:
            self.mult_ranges = {
                str(k): (float(v[0]), float(v[1])) for k, v in state.get("mult_ranges", {}).items()
            }
            self.mult_rates = {str(k): float(v) for k, v in state.get("mult_rates", {}).items()}
            for key in ("mult_min", "mult_max"):
                setattr(self, key, float(state[key]))
        self._mult = {str(k): self._clamp(float(v), str(k)) for k, v in state["multipliers"].items()}
        self._ema = {str(k): float(v) for k, v in state.get("ema", {}).items()}
        if load_hyperparameters:
            self.tolerances = {str(k): float(v) for k, v in state["tolerances"].items()}
            self.default_tolerance = float(state["default_tolerance"])
            for key in ("mult_init", "mult_min", "mult_max", "mult_rate", "mult_ema"):
                setattr(self, key, float(state[key]))

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v:.4g}" for k, v in sorted(self._mult.items()))
        return f"AdaptiveMultipliers({body})"
