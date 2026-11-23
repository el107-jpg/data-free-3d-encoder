"""field_cap: a temporary cap on the eikonal multiplier, released geometrically."""
import json
from types import SimpleNamespace

import pytest

from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.train import Trainer


def _limits(cfg, step):
    return Trainer._field_limits(SimpleNamespace(cfg=cfg), step)


def test_no_cap_by_default():
    assert _limits(DataFreeConfig(), 100) == {}


def test_cap_hold_release_and_off():
    cfg = DataFreeConfig(field_cap=32.0, field_cap_until=8000, field_cap_release_steps=1500, field_cap_release_to=100.0)
    assert _limits(cfg, 5000) == {"eikonal": 32.0}
    assert _limits(cfg, 7999) == {"eikonal": 32.0}
    assert _limits(cfg, 8750)["eikonal"] == pytest.approx((32.0 * 100.0) ** 0.5)
    assert _limits(cfg, 9499)["eikonal"] < 100.0
    assert _limits(cfg, 9500) == {}
    assert "consistency" not in _limits(cfg, 5000)


def test_cap_combines_with_ramp_and_survives_json():
    cfg = DataFreeConfig(field_cap=32.0, field_cap_until=8000, field_ramp_steps=100, field_ramp_start=0.01)
    early = _limits(cfg, 10)
    assert early["eikonal"] < 32.0 and "consistency" in early        # the ramp is lower than the cap early on
    assert _limits(cfg, 500)["eikonal"] == 32.0
    back = DataFreeConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert _limits(back, 500)["eikonal"] == 32.0
