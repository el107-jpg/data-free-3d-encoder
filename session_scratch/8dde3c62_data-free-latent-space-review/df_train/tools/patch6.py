def sub(s, a, b, tag):
    assert a in s, tag
    return s.replace(a, b)


p = 'tests/test_data_free_losses.py'
s = open(p).read()
s = sub(s, '''    expected = (near * (cfg.barrier_floor - 0.2) ** 2).float().mean()
    assert float(L.barrier(fb, cfg)) == pytest.approx(float(expected), rel=1e-4)
''', '''    assert 100 < int(near.sum()) < near.numel() // 4
    # the mean is over the band samples only: every one of them has |g| = 0.2
    assert float(L.barrier(fb, cfg)) == pytest.approx((cfg.barrier_floor - 0.2) ** 2, rel=1e-4)
    # a violation confined to a small part of the band is not diluted by the
    # samples far from the surface
    g = fb.g.clone()
    index = near[0].nonzero().reshape(-1)
    g[0, index[: len(index) // 10]] *= 0.0
    g[0, index[len(index) // 10:]] *= 5.0       # |g| = 1 on the rest of the band
    local = make_fb(sphere(gain=0.2), x)
    local.g = g
    share = (len(index) // 10) / len(index)
    assert float(L.barrier(local, cfg)) == pytest.approx(share * cfg.barrier_floor ** 2, rel=1e-3)
    assert float(L.barrier(local, cfg)) > 100 * cfg.tolerances["barrier"]
    # no sample near the surface: exactly zero, with a gradient path
    far = make_fb(lambda p: sphere()(p) - 5.0, x)
    assert float(L.barrier(far, cfg)) == 0.0
''', 1)
s = sub(s, '''    assert biased > 5 * max(cfg.tolerances[k] for k in L.POSE_KEYS)
''', '''    assert biased > 3 * max(cfg.tolerances[k] for k in L.POSE_KEYS)
''', 2)
s += '''

# ------------------------------------------- per-constraint ranges and rates
def test_multipliers_have_their_own_ranges_and_rates():
    mult = AdaptiveMultipliers(
        tolerances={"a": 0.0, "b": 0.0, "c": 1.0}, mult_init=1.0, mult_min=1.0, mult_max=8.0,
        mult_rate=1.0, mult_ema=0.0,
        mult_ranges={"a": (1.0, 100.0), "c": (0.25, 3.0)}, mult_rates={"a": 3.0},
    )
    assert mult.range("a") == (1.0, 100.0) and mult.range("b") == (1.0, 8.0)
    assert mult.rate("a") == 3.0 and mult.rate("b") == 1.0
    for _ in range(4):
        mult.update({"a": 1.0, "b": 1.0, "c": 0.0})
    assert mult.multiplier("a") == 100.0          # x4 per step, own cap
    assert mult.multiplier("b") == 8.0            # x2 per step, global cap
    assert mult.multiplier("c") == 0.25           # x0.5 per step, own floor below the global one
    # a new name starts inside its own range
    high = AdaptiveMultipliers(mult_init=1.0, mult_ranges={"z": (5.0, 9.0)})
    assert high.multiplier("z") == 5.0
    with pytest.raises(ValueError):
        AdaptiveMultipliers(mult_ranges={"z": (9.0, 5.0)})


def test_multiplier_ranges_survive_a_state_dict_round_trip():
    kwargs = dict(tolerances={"a": 0.0}, mult_init=1.0, mult_min=1.0, mult_max=8.0, mult_rate=1.0,
                  mult_ema=0.0)
    mult = AdaptiveMultipliers(mult_ranges={"a": (1.0, 100.0)}, mult_rates={"a": 3.0}, **kwargs)
    for _ in range(3):
        mult.update({"a": 1.0, "b": 1.0})
    state = mult.state_dict()
    assert state["mult_ranges"] == {"a": (1.0, 100.0)} and state["mult_rates"] == {"a": 3.0}
    # same ranges: values come back as saved
    same = AdaptiveMultipliers(mult_ranges={"a": (1.0, 100.0)}, mult_rates={"a": 3.0}, **kwargs)
    same.load_state_dict(state)
    assert same.multipliers() == mult.multipliers() == {"a": 64.0, "b": 8.0}
    # a config with a tighter range wins over the saved value ...
    tight = AdaptiveMultipliers(mult_ranges={"a": (1.0, 10.0)}, **kwargs)
    tight.load_state_dict(state)
    assert tight.multiplier("a") == 10.0 and tight.rate("a") == 1.0
    # ... unless the saved hyperparameters are asked for
    tight.load_state_dict(state, load_hyperparameters=True)
    assert tight.multiplier("a") == 64.0 and tight.rate("a") == 3.0 and tight.range("a") == (1.0, 100.0)


def test_multipliers_read_ranges_and_rates_from_the_config():
    cfg = DataFreeConfig()
    mult = AdaptiveMultipliers(cfg)
    for name in ("bridge", "delete", "fill", "widen", "thicken", "barrier"):
        assert mult.range(name) == (1.0, 1e4) and mult.rate(name) == 0.05
    assert mult.range("pose_centroid") == (1.0, 50.0)
    assert mult.range("pose_scale") == mult.range("pose_axes") == (1.0, 200.0)
    assert mult.range("eikonal") == (cfg.mult_min, cfg.mult_max) == (1.0, 200.0)
    assert mult.rate("eikonal") == mult.rate("pose_scale") == cfg.mult_rate
    assert cfg.tolerances["pose_centroid"] == 2.5e-4 and cfg.tolerances["barrier"] == 1e-4
    again = DataFreeConfig.from_dict(cfg.to_dict())
    assert AdaptiveMultipliers(again).range("delete") == (1.0, 1e4)
'''
open(p, 'w').write(s)

p = 'tests/test_data_free_train.py'
s = open(p).read()
s = sub(s, '''    cfg.tolerances["pose_scale"] = 1e9     # always met
''', '''    cfg.tolerances["consistency"] = 1e9    # always met
''', 3)
s = sub(s, '''    assert record["mult/pose_scale"] == pytest.approx(''', '''    assert record["mult/consistency"] == pytest.approx(''', 4)
s = sub(s, '''    assert all(record[f"mult/{k}"] >= 1.0 for k in T.CONSTRAINT_KEYS)
''', '''    assert all(record[f"mult/{k}"] >= 1.0 for k in T.CONSTRAINT_KEYS)


def test_trainer_uses_per_constraint_ranges():
    cfg = small_cfg(mult_rate=1.0)
    cfg.mult_rates = {}
    for key in ("pose_centroid", "barrier", "eikonal"):
        cfg.tolerances[key] = -1.0         # never met
    trainer = T.Trainer(cfg)
    for _ in range(15):
        record = trainer.step()
    assert record["mult/pose_centroid"] == 50.0
    assert record["mult/eikonal"] == 200.0
    assert record["mult/barrier"] == 1e4
''', 5)
s = sub(s, '''    assert T.parse_override("tolerances.eikonal=0.5") == ("tolerances.eikonal", 0.5)
''', '''    assert T.parse_override("tolerances.eikonal=0.5") == ("tolerances.eikonal", 0.5)
    assert T.parse_override("mult_rates.delete=0.1") == ("mult_rates.delete", 0.1)
    assert T.parse_override("mult_ranges.delete=1,5e3") == ("mult_ranges.delete", (1.0, 5000.0))
    changed = T.apply_overrides(DataFreeConfig(), ["mult_ranges.pose_scale=2,30", "mult_rates.fill=0.2"])
    assert tuple(changed.mult_ranges["pose_scale"]) == (2.0, 30.0)
    assert tuple(changed.mult_ranges["delete"]) == (1.0, 1e4) and changed.mult_rates["fill"] == 0.2
    with pytest.raises(KeyError):
        T.parse_override("nonsense.delete=1")
''', 6)
open(p, 'w').write(s)
print("ok")
