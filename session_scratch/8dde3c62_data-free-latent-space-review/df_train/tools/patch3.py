p = 'arm_planner/data_free/config.py'
s = open(p).read()
a = '''    mult_min: float = 1e-2
    mult_max: float = 1e4
'''
b = '''    # Range of every multiplier. The split-sample pose terms are unbiased but
    # noisy: at N = 2048 the gradient of pose_centroid is almost pure sampling
    # noise once the centroid is within tolerance (cosine -0.04 between two
    # point samples of the same codes), so a multiplier in the thousands
    # (measured: 3680 with the old cap of 1e4) makes that noise 14 times the
    # diversity gradient. 200 keeps it below the objective. The floor of 1
    # (was 1e-2) keeps a satisfied constraint from needing hundreds of steps
    # to wake up again.
    mult_min: float = 1.0
    mult_max: float = 200.0
'''
assert a in s
open(p, 'w').write(s.replace(a, b))
print("ok")
