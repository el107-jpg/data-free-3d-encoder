def sub(s, a, b, tag):
    assert a in s, tag
    return s.replace(a, b)


p = 'tests/test_data_free_train.py'
s = open(p).read()
s = sub(s, '''    cfg = small_cfg()
    cfg.tolerances["eikonal"] = -1.0       # never met''', '''    cfg = small_cfg(mult_min=1e-2, mult_max=1e4)
    cfg.tolerances["eikonal"] = -1.0       # never met''', 1)
s = sub(s, '''    trainer = T.Trainer(small_cfg(graph_sign_error=-1.0))
    record = trainer.step()
    assert record["graph_mode"] == "off"''', '''    trainer = T.Trainer(small_cfg(graph_sign_error=-1.0, mult_min=1e-2))
    record = trainer.step()
    assert record["graph_mode"] == "off"''', 2)
s = sub(s, '''def test_pose_enters_as_three_separate_constraints():''', '''def test_multipliers_stay_inside_the_default_range():
    cfg = small_cfg()
    assert (cfg.mult_min, cfg.mult_max) == (1.0, 200.0)
    cfg.tolerances["eikonal"] = -1.0       # never met: runs to the cap
    cfg.mult_rate = 1.0                    # x2 per violated step
    trainer = T.Trainer(cfg)
    for _ in range(5):
        record = trainer.step()
    assert record["mult/eikonal"] == 200.0
    assert all(record[f"mult/{k}"] >= 1.0 for k in T.CONSTRAINT_KEYS)


def test_pose_enters_as_three_separate_constraints():''', 3)
open(p, 'w').write(s)
print("ok")
