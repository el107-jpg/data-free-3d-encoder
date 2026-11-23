"""Find Thingi metadata files that no longer read as UTF-8 JSON (the anchor build crashed on one)."""
import glob
import hashlib
import json

D = "/home/eric-luo/Downloads/pointcloud-motion-planning/datasets/thingi10k_source_v1_5_0/prepared_v2/"
bad = []
files = sorted(glob.glob(D + "*.json"))
for path in files:
    raw = open(path, "rb").read()
    try:
        json.loads(raw.decode("utf-8"))
    except Exception as error:
        bad.append((path.split("/")[-1], type(error).__name__, len(raw), hashlib.sha256(raw).hexdigest()[:12]))
print(len(files), "metadata files;", len(bad), "unreadable")
for item in bad[:10]:
    print(" ", item)
if bad:
    raw = open(D + bad[0][0], "rb").read()
    i = next((k for k, b in enumerate(raw) if b > 0x7f), None)
    print("first non-ASCII byte at", i, "context:", raw[max(0, i - 40):i + 20] if i is not None else None)
