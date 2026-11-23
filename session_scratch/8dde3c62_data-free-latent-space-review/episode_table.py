import json

D = "/home/eric-luo/.claude/jobs/8dde3c62/tmp/df_train"
names = ["eikonal", "consistency", "barrier", "pose_centroid", "pose_scale", "pose_axes",
         "bridge", "delete", "fill", "widen", "thicken"]


def table(run, windows):
    rows = {r["step"]: r for r in map(json.loads, open(f"{D}/{run}/metrics.jsonl"))}
    print(f"\n=== {run}: mean of multiplier x value per window")
    head = ["divers", "ratio"] + names + ["walls", "frag0", "gradnorm"]
    print("steps".ljust(13) + "".join(n[:9].rjust(10) for n in head))
    for a, b in windows:
        rs = [rows[s] for s in range(a, b) if s in rows]
        if not rs:
            continue
        m = lambda f: sum(f(r) for r in rs) / len(rs)
        vals = [m(lambda r: r["diversity"]), m(lambda r: r["div/mean_ratio"])]
        vals += [m(lambda r, n=n: r["mult/" + n] * r["term/" + n]) for n in names]
        vals += [m(lambda r: r["graph/wall_sites"]), m(lambda r: r["graph/fragments_kind0"]),
                 m(lambda r: r["grad_norm"])]
        print(f"{a}-{b}".ljust(13) + "".join(f"{v:10.3g}" for v in vals))
    return rows


table("d_resume_cap1e4", [(10000, 10100), (10100, 10300), (10300, 10600), (10600, 11000),
                          (11000, 11500), (11500, 12000)])
rows = table("e", [(6600, 7000), (7000, 7300), (7300, 7400), (7400, 7450), (7450, 7500),
                   (7500, 7550), (7550, 7650), (7650, 7900), (7900, 8300)])
print("\n=== e: multipliers above 1.5 at selected steps")
for s in [6600, 7000, 7300, 7390, 7420, 7450, 7480, 7510, 7540, 7600, 7700, 7900, 8300]:
    r = rows[s]
    print(s, {n: round(r["mult/" + n], 1) for n in names if r["mult/" + n] > 1.5},
          "eik", f'{r["term/eikonal"]:.2g}', "scale", f'{r["term/pose_scale"]:.2g}',
          "thick", f'{r["term/thicken"]:.2g}', "walls", r["graph/wall_sites"])
