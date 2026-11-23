"""Write RUN_INDEX.tsv: one row per run folder, from the run records (no checkpoint is opened)."""
import glob, json, os, time
ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning"
SRC = ROOT + "/runs/data_free_latent"
DEST = ROOT + "/data_free_latent_transfer"
HELPERS = {"analysis_scripts", "monitor", "queue_logs", "review_probes"}

def last_step(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 200000)); tail = f.read().splitlines()
        for line in reversed(tail):
            try:
                return json.loads(line)["step"] + 1
            except Exception:
                continue
    except OSError:
        pass
    return ""

def day(t):
    return time.strftime("%Y-%m-%d", time.localtime(t))

rows = []
for d in sorted(os.listdir(SRC)):
    p = os.path.join(SRC, d)
    if not os.path.isdir(p) or d in HELPERS:
        continue
    files = [os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs]
    pts = [f for f in files if f.endswith(".pt")]
    mt = [os.path.getmtime(f) for f in files] or [os.path.getmtime(p)]
    version = ""
    for name in ("code_version.txt", "launcher/source-commit.txt"):
        q = os.path.join(p, name)
        if os.path.exists(q):
            version = open(q).read().strip().split()[0][:7]
    launchers = sorted(os.path.basename(x) for x in glob.glob(f"{SRC}/queue_logs/*{d}*.sh"))
    rows.append([d, str(last_step(os.path.join(p, "metrics.jsonl"))), day(min(mt)), day(max(mt)),
                 "yes" if os.path.exists(os.path.join(p, "config.json")) else "no",
                 version, ", ".join(launchers), str(len(pts)), f"{sum(os.path.getsize(f) for f in pts) / 1e9:.2f}"])
with open(DEST + "/RUN_INDEX.tsv", "w") as f:
    f.write("run\tupdates_logged\tfirst_file\tlast_file\tconfig_json\trecorded_commit\tlauncher_in_queue_logs\tcheckpoints_left_behind\tcheckpoints_GB\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print(len(rows), "runs")
