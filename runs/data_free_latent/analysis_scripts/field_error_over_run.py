"""The trainer's field error and how often the topology terms were on, by stretch of a run.

usage: field_error_over_run.py <label>=<run folder> [...]
"""
import json
import statistics as st
import sys

for item in sys.argv[1:]:
    label, folder = item.split("=", 1)
    rows = [json.loads(line) for line in open(folder.rstrip("/") + "/metrics.jsonl") if line.strip()]
    rows = [r for r in rows if r["step"] < 1000]
    print(label)
    for a, b in ((0, 100), (100, 250), (250, 500), (500, 750), (750, 1000)):
        seg = [r for r in rows if a <= r["step"] < b]
        e = sorted(r["e"] for r in seg)
        on = [r.get("graph_on", 0) for r in seg]
        print(f"    updates {a:4d}-{b:4d}: field error median {st.median(e):.3f} (10% to 90%: {e[len(e) // 10]:.3f} to {e[9 * len(e) // 10]:.3f}); "
              f"topology terms on {sum(on) / len(on):.0%}; variety loss {st.mean(r['diversity'] for r in seg):.1f}; "
              f"consistency weight {st.mean(r['mult/consistency'] for r in seg):.0f}, slope weight {st.mean(r['mult/eikonal'] for r in seg):.0f}")
