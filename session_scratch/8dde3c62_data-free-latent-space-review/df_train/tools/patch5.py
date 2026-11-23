def sub(s, a, b, tag):
    assert a in s, tag
    return s.replace(a, b)


# ------------------------------------------------------------------ multipliers
p = 'arm_planner/data_free/multipliers.py'
s = open(p).read()
s = sub(s, '''from typing import Dict, Iterable, Mapping, Optional
''', '''from typing import Dict, Iterable, Mapping, Optional, Tuple
''', 1)
s = sub(s, '''rate) once the tolerance is met.  Multipliers are plain Python floats: no
gradient ever flows through them.
"""''', '''rate) once the tolerance is met.  Multipliers are plain Python floats: no
gradient ever flows through them.

Every constraint may have its own range (``mult_ranges[name] = (low, high)``)
and its own rate (``mult_rates[name]``); the global ``mult_min`` / ``mult_max``
/ ``mult_rate`` apply to the names not listed.
"""''', 2)
s = sub(s, '''        hyperparameter is given by keyword.
''', '''        hyperparameter is given by keyword.  ``mult_ranges`` and ``mult_rates``
        (per-constraint overrides) are read from it too when present.
''', 3)
s = sub(s, '''        mult_ema: Optional[float] = None,
        default_tolerance: float = 0.0,
    ) -> None:''', '''        mult_ema: Optional[float] = None,
        default_tolerance: float = 0.0,
        mult_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
        mult_rates: Optional[Mapping[str, float]] = None,
    ) -> None:''', 4)
s = sub(s, '''        if not self.mult_min <= self.mult_max:
            raise ValueError("mult_min must not exceed mult_max")
''', '''        if not self.mult_min <= self.mult_max:
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
''', 5)
s = sub(s, '''    def _clamp(self, value: float) -> float:
        return min(max(value, self.mult_min), self.mult_max)

    def multiplier(self, name: str) -> float:
        """Current multiplier of ``name`` (created at ``mult_init`` if new)."""
        if name not in self._mult:
            self._mult[name] = self._clamp(self.mult_init)
        return self._mult[name]
''', '''    def range(self, name: str) -> Tuple[float, float]:
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
''', 6)
s = sub(s, '''            if ema > self.tolerance(name):
                lam *= 1.0 + self.mult_rate
            else:
                lam *= 1.0 - 0.5 * self.mult_rate
            self._mult[name] = self._clamp(lam)
''', '''            rate = self.rate(name)
            if ema > self.tolerance(name):
                lam *= 1.0 + rate
            else:
                lam *= 1.0 - 0.5 * rate
            self._mult[name] = self._clamp(lam, name)
''', 7)
s = sub(s, '''            "mult_ema": self.mult_ema,
        }
''', '''            "mult_ema": self.mult_ema,
            "mult_ranges": {k: tuple(v) for k, v in self.mult_ranges.items()},
            "mult_rates": dict(self.mult_rates),
        }
''', 8)
s = sub(s, '''        Tolerances and rates stay those of the current config unless
        ``load_hyperparameters`` is set, so a resumed run can change them.
        """
        self._mult = {str(k): self._clamp(float(v)) for k, v in state["multipliers"].items()}
''', '''        Tolerances, ranges and rates stay those of the current config unless
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
''', 9)
s = sub(s, '''                setattr(self, key, float(state[key]))
            self._mult = {k: self._clamp(v) for k, v in self._mult.items()}
''', '''                setattr(self, key, float(state[key]))
''', 10)
open(p, 'w').write(s)

# ----------------------------------------------------------------------- losses
p = 'arm_planner/data_free/losses.py'
s = open(p).read()
a = s[s.index("def barrier(fb: FieldBatch, cfg) -> Tensor:"):s.index("# ---------------------------------------------------------------------- validity")]
b = '''def barrier(fb: FieldBatch, cfg) -> Tensor:
    """Mean of ``relu(barrier_floor - |g|)^2`` over the samples with ``|f| < t/2``.

    The average is over the near-surface samples only (zero when there are
    none), so a local collapse of the gradient is not diluted by the many
    samples far from the surface: the value reads as the typical squared
    shortfall of ``|g|`` in the band.
    """
    near = (fb.f.detach().abs() < 0.5 * float(cfg.min_feature)).to(fb.f.dtype)
    shortfall = torch.relu(float(cfg.barrier_floor) - _norm(fb.g))
    return (near * shortfall.square()).sum() / near.sum().clamp_min(1.0)


'''
s = s.replace(a, b)
open(p, 'w').write(s)

# ----------------------------------------------------------------------- config
p = 'arm_planner/data_free/config.py'
s = open(p).read()
s = sub(s, '''from typing import Any, Dict, Optional
''', '''from typing import Any, Dict, Optional, Tuple
''', 20)
s = sub(s, '''        "barrier": 1e-4,       # mean relu(floor - |g|)^2 near the surface
''', '''        "barrier": 1e-4,       # mean relu(floor - |g|)^2 over the near-surface samples
''', 21)
s = sub(s, '''        "pose_centroid": 1e-4,  # |centroid|^2: centroid within 0.01
''', '''        # |centroid|^2: centroid within 0.016. Its sampling noise is 1e-3 per
        # shape at N = 2048, so 1e-4 kept the multiplier pinned on noise.
        "pose_centroid": 2.5e-4,
''', 22)
s = sub(s, '''@dataclass
class DataFreeConfig:''', '''ACTION_CONSTRAINTS = ("bridge", "delete", "fill", "widen", "thicken")


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
    ranges["pose_centroid"] = (1.0, 50.0)
    ranges["pose_scale"] = (1.0, 200.0)
    ranges["pose_axes"] = (1.0, 200.0)
    return ranges


def default_mult_rates() -> Dict[str, float]:
    """Per-constraint multiplier rate; names not listed use mult_rate.

    0.05 takes a multiplier from 1 to 1e3 in about 140 violated steps.
    """
    rates = {name: 0.05 for name in ACTION_CONSTRAINTS}
    rates["barrier"] = 0.05
    return rates


@dataclass
class DataFreeConfig:''', 23)
s = sub(s, '''    mult_rate: float = 0.02
    mult_ema: float = 0.95
''', '''    mult_rate: float = 0.02
    mult_ema: float = 0.95
    mult_ranges: Dict[str, Tuple[float, float]] = field(default_factory=default_mult_ranges)
    mult_rates: Dict[str, float] = field(default_factory=default_mult_rates)
''', 24)
s = sub(s, '''    # Range of every multiplier. The split-sample pose terms are unbiased but''',
        '''    # Range of every multiplier not listed in mult_ranges. The split-sample pose terms are unbiased but''', 25)
open(p, 'w').write(s)

# ------------------------------------------------------------------------ train
p = 'arm_planner/data_free/train.py'
s = open(p).read()
s = sub(s, '''    if "." in key:
        head, _ = key.split(".", 1)
        if head != "tolerances":
            raise KeyError(f"unknown config field {key!r}")
        return key, float(text)
''', '''    if "." in key:
        head, _ = key.split(".", 1)
        if head in ("tolerances", "mult_rates"):
            return key, float(text)
        if head == "mult_ranges":  # mult_ranges.delete=1,1e4
            low, high = (float(part) for part in text.split(","))
            return key, (low, high)
        raise KeyError(f"unknown config field {key!r}")
''', 30)
s = sub(s, '''    ``key`` is a :class:`DataFreeConfig` field, or ``tolerances.<name>`` for
    one entry of the tolerance table.
''', '''    ``key`` is a :class:`DataFreeConfig` field, or one entry of a table:
    ``tolerances.<name>=x``, ``mult_rates.<name>=x``, ``mult_ranges.<name>=low,high``.
''', 31)
open(p, 'w').write(s)
print("ok")
