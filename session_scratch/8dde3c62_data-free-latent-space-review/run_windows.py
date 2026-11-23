import json
import sys

run, width = sys.argv[1], int(sys.argv[2])
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
names = ["eikonal", "consistency", "barrier", "pose_centroid", "pose_scale", "pose_axes",
         "bridge", "delete", "fill", "widen", "thicken"]
rows = [json.loads(l) for l in open(f"/home/eric-luo/.claude/jobs/8dde3c62/tmp/df_train/{run}/metrics.jsonl")]
rows = [r for r in rows if r["step"] >= start]
print("window".ljust(11) + "".join(n[:9].rjust(10) for n in ["ratio", "deadzone", "eik"] + names + ["frag0", "gaps", "walls", "cav", "gnorm"]))
for a in range(rows[0]["step"] // width * width, rows[-1]["step"] + 1, width):
    rs = [r for r in rows if a <= r["step"] < a + width]
    if not rs:
        continue
    m = lambda f: sum(f(r) for r in rs) / len(rs)
    vals = [m(lambda r: r["div/mean_ratio"]), m(lambda r: r["div/dead_zone_fraction"]), m(lambda r: r["term/eikonal"])]
    vals += [m(lambda r, n=n: r["mult/" + n] * r["term/" + n]) for n in names]
    vals += [m(lambda r: r["graph/fragments_kind0"]), m(lambda r: r["graph/gap_sites"]), m(lambda r: r["graph/wall_sites"]),
             m(lambda r: r["graph/cavities"]), m(lambda r: r["grad_norm"])]
    print(f"{a}".ljust(11) + "".join(f"{v:10.3g}" for v in vals))
last = rows[-1]
print("multipliers now:", {n: round(last["mult/" + n], 1) for n in names if last["mult/" + n] > 1.05})
