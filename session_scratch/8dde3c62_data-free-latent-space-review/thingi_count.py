"""How many prepared Thingi10K originals are clean, single-piece, hole-free solids by their metadata?"""
import glob
import json
from collections import Counter

D = "/home/eric-luo/Downloads/pointcloud-motion-planning/datasets/thingi10k_source_v1_5_0/prepared_v2/"
files = sorted(glob.glob(D + "*.json"))
keep, why = [], Counter()
for path in files:
    meta = json.load(open(path))
    g = meta.get("source_geometry_metadata", {})

    def val(key, default="nan"):
        try:
            return float(g.get(key, default))
        except (TypeError, ValueError):
            return float("nan")

    checks = {
        "one component": val("num_connected_components") == 1,
        "genus 0": val("euler_characteristic") == 2,
        "closed": val("num_boundary_edges") == 0,
        "manifold": val("vertex_manifold") == 1 and val("edge_manifold") == 1,
        "oriented": val("oriented") == 1,
        "no self-intersection": val("num_self_intersections") == 0,
        "solid": val("solid", "1") == 1,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        for k in failed:
            why[k] += 1
    else:
        keep.append((meta["file_id"], int(val("num_faces")), meta.get("name", ""), meta.get("category", "")))
print(f"{len(files)} prepared originals; {len(keep)} pass every metadata check")
print("failures by check (an object can fail several):", dict(why))
faces = sorted(k[1] for k in keep)
print("faces of the kept: min", faces[0], "median", faces[len(faces) // 2], "p90", faces[int(0.9 * len(faces))], "max", faces[-1])
print("categories:", Counter(k[3] for k in keep).most_common(12))
json.dump([k[0] for k in keep], open("/home/eric-luo/.claude/jobs/8dde3c62/tmp/thingi_candidates.json", "w"))
