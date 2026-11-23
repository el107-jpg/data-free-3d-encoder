"""Training-log trend of a run by stretches of updates.   usage: run_trend.py <run folder> <start> <end> <width>"""
import json
import statistics as st
import sys

folder, start, end, width = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = [json.loads(line) for line in open(folder.rstrip("/") + "/metrics.jsonl") if line.strip()]
for a in range(start, end, width):
    seg = [r for r in rows if a <= r["step"] < a + width]
    if not seg:
        continue
    off = sum(r["graph_mode"] == "off" for r in seg) / len(seg)
    print(f"  {a:5d}-{a + width:5d}: field error {st.median(r['e'] for r in seg):.3f}; topology off {off:.0%}; "
          f"variety loss {st.mean(r['diversity'] for r in seg):.2f}; anchor overlap in batch {st.mean(r.get('anchor/iou', float('nan')) for r in seg):.3f}; "
          f"anchor term {st.mean(r.get('anchor/term', float('nan')) for r in seg):.4f}; slope weight {st.mean(r['mult/eikonal'] for r in seg):.0f}; "
          f"thin-wall sites {st.mean(r.get('graph/wall_sites', 0) for r in seg):.0f}")
