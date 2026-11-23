import shutil, sys

p = "/home/eric-luo/Downloads/pointcloud-motion-planning/Data-free latent space.md"
bak = "/home/eric-luo/.claude/jobs/8dde3c62/tmp/note_before_continuity.md"
s = open(p, encoding="utf-8").read()
if "8. Keeping transformations continuous" in s:
    sys.exit("already applied")

edits = [
    # 1. new section before the closing training paragraph
    ("Training this whole thing is going to be quite complicated, so there are",
     "8. Keeping transformations continuous along the sphere\n\n"
     "Topology can only change when two facing walls meet, so we never let walls closer than a minimum feature size `t` (at least a couple of sample spacings) persist. Every code starts as a sphere, so every shape stays one piece with no cavities and no handles, and thin spikes and sheets cannot vanish by thinning, which is what would make the distance field jump between nearby codes. Enforcement is by the graph (widen thin gaps within one piece, thicken thin walls; see the actions table below), plus a free barrier loss on existing samples: where `|f| < t/2`, penalise `relu(0.5 − |∇f|)²`. With this in place the field must be a true distance everywhere, and we count handles on anchor codes with a coarse voxel grid now and then as a check.\n\n"
     "Training this whole thing is going to be quite complicated, so there are"),
    # 2. two more rows in the actions table
    ("## Two graphs, three actions", "## Two graphs, five actions"),
    ("| Enclosed free region in the optimistic outside graph | Fill | It is enclosed for real |\n",
     "| Enclosed free region in the optimistic outside graph | Fill | It is enclosed for real |\n"
     "| Gap narrower than `t` between two parts of the same piece | Widen: push the free samples in the gap to `−t/2` or below | Bridging here would create a handle |\n"
     "| Wall thinner than `t` (found in the outside graph) | Thicken: push the inside samples in the wall to `t/2` or above | Stops punctures and parts vanishing by thinning |\n"),
    # 3. tolerance policy
    ("Gaps thinner than the resolution are treated as connected. The resolution is\n"
     "roughly `2δ` plus the local sample spacing. A gap the robot cannot enter does\n"
     "not matter to the planner.\n\n"
     "Merging in the graph does not close the gap in the field. If the single-piece\n"
     "criterion in `scripts/check_bar.py` counts pieces by sign on a grid, such\n"
     "fragments will still count as separate. In that case, also apply the bridging\n"
     "loss across merged gaps. To be checked before the first run.\n",
     "Gaps thinner than the resolution cannot be told apart from contact. The\n"
     "resolution is roughly `2δ` plus the local sample spacing, and the minimum\n"
     "feature size `t` (section 8) must be at least this.\n\n"
     "A thin gap between two different pieces is treated as connected and gets the\n"
     "bridging loss, so it really closes. A thin gap between two parts of the same\n"
     "piece is never bridged, because that would create a handle; it is widened\n"
     "instead.\n"),
]
m = s
for old, new in edits:
    if m.count(old) != 1:
        sys.exit("anchor not found exactly once: " + old[:60])
    m = m.replace(old, new)
shutil.copyfile(p, bak)
open(p, "w", encoding="utf-8").write(m)
print("applied", len(s), "->", len(m))
