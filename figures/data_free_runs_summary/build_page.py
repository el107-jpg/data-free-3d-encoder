import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from build_data import ING, PHASES, E, POTENTIAL
from PIL import Image

FIG = "/home/eric-luo/Downloads/pointcloud-motion-planning/figures"
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, "img"), exist_ok=True)

total = 0
for ent in E:
    imgs = []
    for name, cap in ent["imgs"]:
        src = os.path.join(FIG, name + ".png")
        if not os.path.exists(src):
            print("MISSING", src); continue
        dst = os.path.join(OUT, "img", name + ".jpg")
        if not os.path.exists(dst):
            im = Image.open(src).convert("RGB")
            im.thumbnail((1400, 1400))
            im.save(dst, quality=80, optimize=True, progressive=True)
        total += os.path.getsize(dst)
        imgs.append({"f": "img/" + name + ".jpg", "c": cap, "src": src})
    ent["imgs"] = imgs
print("images MB", round(total / 1e6, 1))

data = {"ing": [{"k": k, "l": l, "s": s} for k, l, s in ING],
        "phases": [{"k": k, "d": d, "t": t, "s": s} for k, d, t, s in PHASES],
        "E": E,
        "pot": [{"lvl": a, "t": b, "ing": c, "why": d} for a, b, c, d in POTENTIAL]}
tpl = open(os.path.join(os.path.dirname(__file__), "template.html")).read()
html = tpl.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
open(os.path.join(OUT, "data_free_runs.html"), "w").write(html)
files = {f"img/{fn}": os.path.join(OUT, "img", fn) for fn in sorted(os.listdir(os.path.join(OUT, "img")))}
json.dump(files, open(os.path.join(OUT, "files.json"), "w"), indent=0)
print(len(files), "files; html KB", os.path.getsize(os.path.join(OUT, "data_free_runs.html")) // 1024)
