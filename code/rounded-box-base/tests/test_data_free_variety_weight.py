"""The variety-weight controller: grows only when safe, shrinks otherwise, stays in range, off by default."""
from types import SimpleNamespace

import pytest

from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.multipliers import AdaptiveMultipliers
from arm_planner.data_free.train import Trainer


class Fake(SimpleNamespace):
    _variety_weight_now = Trainer._variety_weight_now
    _adapt_variety_weight = Trainer._adapt_variety_weight


def _fake(consistency=0.02, widen=0.0, **kw):
    cfg = DataFreeConfig(accumulate=4, variety_weight_max=4.0, **kw)
    mult = AdaptiveMultipliers(cfg)
    mult._ema.update({"consistency": consistency, "bridge": 0.0, "delete": 0.0, "fill": 0.0, "widen": widen})
    return Fake(cfg=cfg, multipliers=mult)


def test_off_by_default():
    assert Fake(cfg=DataFreeConfig(variety_weight=1.5), _variety_weight=3.0)._variety_weight_now() == 1.5


def test_grows_when_safe_and_stops_at_the_limit():
    t = _fake()
    assert t._adapt_variety_weight(True) and t._variety_weight == pytest.approx(1.001)
    for _ in range(2000):
        t._adapt_variety_weight(True)
    assert t._variety_weight == 4.0


@pytest.mark.parametrize("kw,measured", [(dict(consistency=0.03), True), (dict(widen=1.0), True), (dict(), False)])
def test_shrinks_when_unsafe_and_not_below_the_base(kw, measured):
    t = _fake(**kw)
    t._variety_weight = 2.0
    assert not t._adapt_variety_weight(measured)
    assert t._variety_weight == pytest.approx(2.0 * 0.998)
    for _ in range(2000):
        t._adapt_variety_weight(measured)
    assert t._variety_weight == 1.0


def test_bad_settings_rejected():
    with pytest.raises(ValueError):
        DataFreeConfig(variety_weight_max=4.0, accumulate=1)
    with pytest.raises(ValueError):
        DataFreeConfig(variety_weight=5.0, variety_weight_max=4.0, accumulate=4)
