import json, glob, os
import numpy as np
D = "/home/eric-luo/.claude/jobs/8dde3c62/tmp/"
for p in sorted(glob.glob(D + "shell_variance_*.json")):
    j = json.load(open(p))
    loc = np.sqrt(np.array(j["local_pose_removed_variances"]))
    glo = np.sqrt(np.maximum(np.array(j["global_pose_removed_variances"]), 0))
    tag = os.path.basename(p)[len("shell_variance_"):-5]
    print(f"== {tag}  step {j['step']}  eff local {j['local_pose_removed']['effective_directions']:.2f} total {j['local_pose_removed']['total']:.3f} "
          f"eff global {j['global_pose_removed']['effective_directions']:.2f}  deform share {j['deformation_share_of_motion']:.2f}")
    print("   local RMS :", " ".join(f"{v:.3f}" for v in loc[:12]), "| 16:", f"{loc[15]:.4f}", "31:", f"{loc[30]:.4f}")
    print("   global RMS:", " ".join(f"{v:.3f}" for v in glo[:12]), "| 16:", f"{glo[15]:.4f}", "31:", f"{glo[30]:.4f}")
    r = loc[:-1] / np.maximum(loc[1:], 1e-12)
    k = int(np.argmax(r[:15])) + 1
    print(f"   biggest local gap after direction {k}: ratio {r[k-1]:.1f};  n(local RMS>0.05)={int((loc>0.05).sum())}  n(>0.02)={int((loc>0.02).sum())} n(>0.01)={int((loc>0.01).sum())}")
