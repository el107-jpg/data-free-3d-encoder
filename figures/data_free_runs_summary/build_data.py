"""Experiment history of the data-free latent work, as data for the summary page."""

ING = [  # key, label, short
    ("D32", "32-D codes", "32-D"),
    ("D64", "64-D codes", "64-D"),
    ("SPR", "Spread term", "Spread"),
    ("ORD", "Ordinal distances", "Ordinal"),
    ("STAT", "Shape-statistic push", "Stats"),
    ("NN", "Nearest-neighbour variety", "Nearest"),
    ("WEAK", "Weak pairs / frame", "Weak/frame"),
    ("ANCT", "Thingi anchors", "Thingi anc."),
    ("ANCP", "Procedural anchors", "Proc. anc."),
    ("THIN", "Thin anchors in set", "Thin anc."),
    ("TRAIN", "Trainable anchor codes", "Train codes"),
    ("TOL8", "Consistency tol. 0.008", "Tol 0.008"),
    ("SOFT", "Weakened consistency, uncapped", "Soft cons."),
    ("CAP", "Capped-follow consistency", "Capped"),
    ("RAMP", "Field-weight ramp", "Ramp"),
    ("NOC", "No filled centre", "No centre"),
    ("EXT", "Extent pose rule (p=6)", "Extent"),
    ("EVEN", "Even-code penalty", "Even code"),
    ("LEAK", "Leaky activation", "Leak"),
    ("LN", "Layer norm", "Layer norm"),
    ("SHARP", "Sharp activation (β 100)", "β 100"),
    ("FOUR", "Fourier bands", "Fourier"),
    ("BAND", "Surface-band variety", "Surf. band"),
    ("NOTHK", "No thicken term", "No thicken"),
    ("LONG", "Long run (5k+ updates)", "Long"),
    ("SETTLE", "Low-LR settle stage", "Settle"),
    ("CREASE", "Crease (flat-face) term", "Crease"),
    ("WN", "Weight norm", "Weight norm"),
    ("OUTLR", "Final layer at 0.1× learning rate", "Out LR 0.1"),
    ("NOBASE", "No fixed sphere base", "No base"),
    ("BOX", "Learned rounded-box base", "Box base"),
    ("CORR", "Cost on correction size", "Corr. cost"),
]

PHASES = [
    ("p1", "9 Nov", "Getting the trainer to behave",
     "Sphere-plus-correction decoder, constraint multipliers, topology graph. The first runs either stayed spheres or grew loose blobs."),
    ("p2", "10 Nov", "Variety objectives, no anchors (32-D)",
     "Different ways of pushing shapes apart. Each found a few dials of variation and then stopped adding directions."),
    ("p3", "11 Nov", "Anchor objects",
     "A few real solids reproduced at fixed codes keep more code directions alive. Settings around them were tuned: pull, neighbourhood, tolerance, code size."),
    ("p4", "11–12 Nov", "Easing the consistency brake",
     "Consistency fought variety (gradient cosine −0.95). Uncapped weakening collapsed; capped-follow worked."),
    ("p5", "12 Nov", "No filled centre, procedural anchors",
     "Letting shapes leave the centre empty made thin shapes reachable. Anchors chosen by kind of geometry, and a failed pose rule."),
    ("p6", "12–13 Nov", "Using every code direction, network health",
     "Even-code penalty, dead units, leak or layer norm, and capacity tests of the decoder."),
    ("p7", "13–14 Nov", "Current recipe and refinements",
     "J4 (96 anchors, β 100, Fourier 3) and its continuations, trainable codes, the surface band, the crease term, weight norm."),
    ("p9", "14–15 Nov", "Removing the fixed sphere",
     "Could the decoder do better without the sphere under every shape? After a field flaw was found and fixed it trains, but it stays behind J4."),
    ("p10", "14–16 Nov", "Learned rounded-box base",
     "A learned head gives each code its own rounded box (rod, plate, block) as the base. The head only gets used from scratch, then fades unless pushed; the latest runs add a cost on the correction."),
    ("p8", "—", "Ideas built or discussed but never trained",
     "Kept so they are not re-proposed without knowing their history."),
]

# verdict: helped, mixed, nogain, hurt, failed, diag, running
E = []
def e(**k):
    k.setdefault("imgs", []); k.setdefault("ing", []); k.setdefault("m", {})
    E.append(k)

# ---------------- phase 1
e(id="a", ph="p1", date="9 Nov", title="Smoke test", verdict="diag",
  idea="300 updates of the new trainer to measure speed and memory.",
  out="208 ms per update, 7.5 GB peak; the graph build is about 4% of the step.")
e(id="b", ph="p1", date="9 Nov", title="Single pose loss", verdict="failed",
  idea="Original single pose-pinning loss at tolerance 1e-4 (and a looser 2e-3).",
  out="At 1e-4 the multiplier ran to its cap and everything became a sphere. At 2e-3 shapes were resized, barely dented spheres; nearly all surface motion was pose.",
  imgs=[("data_free_latent_slices_pose_tol_1e-4", "Pose tolerance 1e-4: every code a sphere"),
        ("data_free_latent_slices", "Pose tolerance 2e-3: dented spheres")], ing=["D32"])
e(id="c", ph="p1", date="9 Nov", title="Split pose constraints", verdict="helped",
  idea="Pose split into three unbiased constraints (centre, size, axes).",
  out="First real deformation: pose share of motion fell from about 100% to 35%, all probes one piece, thin-wall terms fired for the first time.",
  imgs=[("data_free_latent_slices_c", "Run c: first real deformation")], ing=["D32"])
e(id="d", ph="p1", date="9 Nov", title="Long run, one multiplier cap", verdict="failed",
  idea="Run c continued to 10,000 updates, then the cap raised to 1e4.",
  out="From about update 5,500 every shape grew 2–4 detached blobs. With the higher cap the blobs went but pose multipliers ran away and variety collapsed. Led to per-constraint multiplier ranges.",
  imgs=[("data_free_latent_slices_d", "Run d: detached blobs at the edge")], ing=["D32", "LONG"])
e(id="e", ph="p1", date="9 Nov", title="Per-constraint ranges", verdict="failed",
  idea="Separate multiplier ranges per constraint, barrier averaged near the surface only.",
  out="Blobs were cleaned up within 70–190 updates, but each clean-up reset shapes to near-spheres. The final model was 77% resizing.",
  imgs=[("data_free_latent_slices_e", "Run e: clean but sphere-like after each reset")], ing=["D32", "LONG"])
e(id="clean", ph="p1", date="10 Nov", title="Which repair breaks shapes?", verdict="diag",
  idea="Three 900-update continuations of the blob model: all repairs, no thicken, thicken only.",
  out="The thicken term caused the blob-and-reset cycle. Fix: thin sites counted on the main body only, and a looser thicken tolerance.",
  imgs=[("data_free_clean_all_actions", "All repair terms on"), ("data_free_clean_no_thicken", "Thicken held off"),
        ("data_free_clean_thicken_only", "Thicken only")])

# ---------------- phase 2
e(id="g", ph="p2", date="10 Nov", title="Pose-invariant variety", verdict="mixed",
  idea="Variety distance ignores small moves, turns and resizes; slope mode, 32 shapes per update.",
  out="Pose leak gone (93% real deformation), clean topology, but a single dial: 99.9% of variance in one squash/stretch direction.",
  m=dict(var=0.221), imgs=[("data_free_shapes_pose_invariant_final", "One dial: squash and stretch")], ing=["D32"])
e(id="h2", ph="p2", date="10 Nov", title="64 shapes per update", verdict="mixed",
  idea="Gradient accumulation to 64 shapes per update.",
  out="Three directions alive instead of one, first real limbs; validity slipped to 81% at the end.",
  m=dict(var=0.275), imgs=[("data_free_shapes_acc64_final", "64 shapes per update: first limbs")], ing=["D32"])
e(id="i", ph="p2", date="10 Nov", title="128 shapes per update", verdict="nogain",
  idea="Accumulation to 128 shapes per update, same total shapes seen.",
  out="No extra directions (effective 1.9); fully clean.", m=dict(var=0.195),
  imgs=[("data_free_shapes_acc128_final", "128 shapes per update")], ing=["D32"])
e(id="j", ph="p2", date="10 Nov", title="Spread term", verdict="mixed",
  idea="Different code directions in a cluster must change the shape in different ways.",
  out="Effective directions 7.2 (from 2.1), but variety halved: lumpy spheres, no limbs.",
  m=dict(var=0.118), imgs=[("data_free_shapes_spread64_final", "Spread: many weak directions, lumpy spheres")], ing=["D32", "SPR", "LONG"])
e(id="k", ph="p2", date="10 Nov", title="Ordinal distances", verdict="failed",
  idea="Only the order of distances is enforced above a floor.",
  out="Collapsed to one steep dial; 56% of probes clean, detached pieces and handles.",
  m=dict(var=4.84), imgs=[("data_free_shapes_ordinal64_final", "Ordinal: one steep dial")], ing=["D32", "ORD", "LONG"])
e(id="m", ph="p2", date="10 Nov", title="Ordinal + spread", verdict="helped",
  idea="Ordinal floor and ordering, plus the spread term.",
  out="Best cluster run: effective directions 6.4, 7 directions above 0.05, clean at all 21 checks. Still stopped at 7 directions.",
  m=dict(var=0.24), imgs=[("data_free_shapes_ordinal_spread64_final", "Ordinal + spread")], ing=["D32", "ORD", "SPR", "LONG"])
e(id="n", ph="p2", date="10 Nov", title="Phase 2 shape statistics", verdict="nogain",
  idea="Ten shape descriptors (thickness, elongation, …) added to the pair distance.",
  out="Variety fell (effective 5.4). The statistics were half the gradient but noisy and nearly orthogonal to the field term.",
  m=dict(var=0.17), imgs=[("data_free_shapes_phase2_final", "Phase 2 statistics")], ing=["D32", "ORD", "SPR", "STAT", "LONG"])
e(id="o", ph="p2", date="10 Nov", title="Statistic repulsion", verdict="mixed",
  idea="Each statistic group (proportions, thickness, compactness) pushed apart on its own.",
  out="Statistic spreads rose 18–32%, but no new kinds of shape and more thin walls.",
  m=dict(var=0.20), imgs=[("data_free_shapes_stat_repulsion_final", "Statistic repulsion")], ing=["D32", "ORD", "SPR", "STAT"])
e(id="p", ph="p2", date="10 Nov", title="Pure log repulsion", verdict="hurt",
  idea="A plain −log push on the latent-weighted mean distance, plus spread.",
  out="Variance piled into two dials (plates and rods); 88% clean, many thin walls.",
  m=dict(var=0.66), imgs=[("data_free_shapes_repel_spread64_final", "Repel: plates and rods")], ing=["D32", "SPR", "LONG"])
e(id="q", ph="p2", date="10 Nov", title="Local Chamfer push", verdict="nogain",
  idea="Push surfaces apart where they are locally far, point-to-plane.",
  out="Local gap +6% only, no new local features. Cheap and clean.", m=dict(var=0.20),
  imgs=[("data_free_shapes_chamfer_look_final", "Local Chamfer")], ing=["D32", "ORD", "SPR"])
e(id="r", ph="p2", date="10 Nov", title="Global nearest-neighbour", verdict="helped",
  idea="No clusters: random codes, each shape pushed from its nearest neighbours.",
  out="Variance 0.38 (+60%), the most distinct shapes so far (wedges, boomerangs). Became the base objective for everything after.",
  m=dict(var=0.38), imgs=[("data_free_shapes_global_nearest64_final", "Nearest-neighbour repulsion")], ing=["D32", "NN", "LONG"])
e(id="s", ph="p2", date="10 Nov", title="Neighbour pool", verdict="failed",
  idea="Nearest-neighbour search against a pool of 256 extra codes.",
  out="Machine crash at about update 740; never resumed. Promising at update 500 (9 live directions, all clean).", ing=["D32", "NN"])
e(id="t", ph="p2", date="10–11 Nov", title="Batch-wide neighbours", verdict="mixed",
  idea="Every shape compared with all 127 others of the update, exact gradient.",
  out="128/128 clean, but the direction count froze at about 9–10 from update 500 on, then a 100× cliff.",
  m=dict(var=0.287, dirs=9, clean=122), imgs=[("data_free_shapes_noanchors4500", "No anchors, update 4,500")], ing=["D32", "NN", "LONG"])
e(id="u", ph="p2", date="11 Nov", title="Weak-direction pairs", verdict="nogain",
  idea="Pairs of codes that differ only in unused directions, pushed apart.",
  out="Existing axes just rotated into the unused set. Still 7 strong directions.",
  m=dict(var=0.214, clean=126), imgs=[("data_free_shapes_weakpairs3000", "Weak pairs")], ing=["D32", "NN", "WEAK"])
e(id="v", ph="p2", date="11 Nov", title="Log-det frame", verdict="nogain",
  idea="Log-volume of a frame of partner codes around one code.",
  out="Evened the 10 existing directions but created none.",
  m=dict(var=0.196, clean=126), imgs=[("data_free_shapes_frame3000", "Frame term")], ing=["D32", "NN", "WEAK"])

# ---------------- phase 3
e(id="w", ph="p3", date="11 Nov", title="First anchors", verdict="mixed",
  idea="64 Thingi10K solids reproduced at fixed perpendicular codes.",
  out="Kept the weak tail of directions alive (11–16 above 0.01) and anchors reached 0.91. Variance was flat and random codes were bland potato blobs.",
  m=dict(var=0.084, clean=128, anch=0.91),
  imgs=[("data_free_anchor_objects", "The 64 Thingi anchors"), ("data_free_random_shapes_ref2500", "Random codes, update 2,500"),
        ("data_free_anchor_fit_ref2500", "Anchor reproductions")], ing=["D32", "NN", "ANCT", "TRAIN"])
e(id="x", ph="p3", date="11 Nov", title="Weak pull + frame", verdict="hurt",
  idea="Anchor pull at weight 10 plus a full frame pass.",
  out="Worse on everything: variance 0.028, anchors 0.59, ridged 'tennis-ball' spheres.",
  m=dict(var=0.028, anch=0.59), imgs=[("data_free_random_shapes_varA1000", "Tennis-ball spheres")], ing=["D32", "NN", "ANCT", "TRAIN", "WEAK"])
e(id="y", ph="p3", date="11 Nov", title="Anchor neighbourhood", verdict="mixed",
  idea="Anchor pull applied around each anchor's code with a Gaussian falloff (radius 0.5).",
  out="Directions more even sooner (14 above 0.01); random codes still bland. Radius 0.5 kept in every later run.",
  m=dict(var=0.068, anch=0.80), imgs=[("data_free_random_shapes_varB1000", "Neighbourhood pull")], ing=["D32", "NN", "ANCT", "TRAIN"])
e(id="z", ph="p3", date="11 Nov", title="Wider neighbourhood", verdict="nogain",
  idea="Radius 0.8, so random codes are pulled too.", out="Same as radius 0.5.",
  m=dict(var=0.069, anch=0.77), ing=["D32", "NN", "ANCT", "TRAIN"])
e(id="d1", ph="p3", date="11 Nov", title="Weaker pull (30)", verdict="mixed",
  idea="Neighbourhood pull at weight 30.", out="Variance +65% and more angular shapes, but the weak tail faded.",
  m=dict(var=0.112, clean=128, anch=0.78), imgs=[("data_free_random_shapes_d1_1000", "Pull 30")], ing=["D32", "NN", "ANCT", "TRAIN"])
e(id="d2", ph="p3", date="11 Nov", title="Anchors exempt from field terms", verdict="helped",
  idea="No eikonal or consistency at anchor codes.", out="Small even gain; consistency multiplier came off its cap. Kept since.",
  m=dict(var=0.076, anch=0.81), ing=["D32", "NN", "ANCT", "TRAIN"])
e(id="L2", ph="p3", date="11 Nov", title="Long anchor run, 32-D", verdict="helped",
  idea="Radius 0.5, exemption, pull 100, 5,000 updates (L1: pull 30).",
  out="128/128 clean, anchors 0.88; unseen-object fit 0.875 against 0.788 without anchors. Became the 32-D base. L1 (pull 30) had more variance (0.20) but weaker fits.",
  m=dict(var=0.155, dirs=13, clean=128, anch=0.88),
  imgs=[("data_free_random_shapes_L2_5000", "L2 random codes"), ("data_free_heldout_fits", "Unseen-object fits")],
  ing=["D32", "NN", "ANCT", "TRAIN", "LONG"])
e(id="F2s", ph="p3", date="11 Nov", title="Settle stage", verdict="helped",
  idea="L2 plus 300 updates at one-tenth learning rate.", out="Variance 0.153 → 0.193, anchors 0.88 → 0.95, multipliers steadier.",
  m=dict(var=0.193, dirs=14, clean=128, anch=0.95), ing=["D32", "NN", "ANCT", "TRAIN", "LONG", "SETTLE"])
e(id="F3", ph="p3", date="11 Nov", title="Consistency tolerance 0.008", verdict="helped",
  idea="L2 continued with the consistency tolerance loosened from 0.005 to 0.008 (control: unchanged).",
  out="Variance 0.258, +71% over the control; one handle; thin walls doubled.",
  m=dict(var=0.258, dirs=14, clean=127, anch=0.88), ing=["D32", "NN", "ANCT", "TRAIN", "LONG", "TOL8"])
e(id="F5", ph="p3", date="11 Nov", title="Seed repeats", verdict="diag",
  idea="L2 recipe with two more seeds, 1,000 updates.", out="Noise is about 8% in variance and one direction.")
e(id="F4", ph="p3", date="11 Nov", title="Smaller inner ball", verdict="nogain",
  idea="Inner ball 0.15 → 0.05 with a larger anchor pool.", out="No clear variety gain; the thinnest rod anchor failed.",
  m=dict(var=0.133, dirs=12, clean=128, anch=0.78), imgs=[("data_free_random_shapes_F4_inner005_2500", "Inner ball 0.05")], ing=["D32", "NN", "ANCT", "TRAIN"])
e(id="F6", ph="p3", date="11 Nov", title="64-D codes", verdict="helped",
  idea="Latent size 64, each anchor on its own axis.", out="Variance 0.143 at 1,000 against 0.08 for 32-D; only about 16 of 64 directions used.",
  m=dict(var=0.175, dirs=13, clean=128, anch=0.87), imgs=[("data_free_random_shapes_F6_dim64_2500", "64-D codes")], ing=["D64", "NN", "ANCT", "TRAIN"])
e(id="G1", ph="p3", date="11 Nov", title="64-D + tol 0.008, trainable codes", verdict="helped",
  idea="Both gains together from scratch, anchor codes trainable.", out="Variance 0.218, 15 directions.",
  m=dict(var=0.218, dirs=15, clean=128, anch=0.87), ing=["D64", "NN", "ANCT", "TRAIN", "TOL8"])
e(id="G2", ph="p3", date="11 Nov", title="Frozen anchor codes", verdict="helped",
  idea="G1 with anchor codes frozen.", out="Variance 0.327 (+50% over G1). Became the base.",
  m=dict(var=0.327, dirs=16, clean=128, anch=0.86), imgs=[("data_free_random_shapes_G2_dim64_tol008_frozen_2500", "G2 random codes")],
  ing=["D64", "NN", "ANCT", "TOL8"])
e(id="G3", ph="p3", date="11 Nov", title="G2 to 5,000 (+ settle)", verdict="mixed",
  idea="G2 continued to 5,000, then 300 settle updates (G3s).",
  out="Variance 0.41 with thin walls and 4 bad codes; the settle stage repaired validity (128/128) and gave back variance (0.32).",
  m=dict(var=0.41, dirs=15, clean=124, anch=0.89), imgs=[("data_free_random_shapes_G3_dim64_tol008_frozen_5000", "G3 at 5,000")],
  ing=["D64", "NN", "ANCT", "TOL8", "LONG", "SETTLE"])

# ---------------- phase 4
e(id="S2", ph="p4", date="11 Nov", title="Fixed-surface consistency", verdict="failed",
  idea="Consistency may not move the surface (landing point with frozen weights).", out="Shapes grew, then collapsed to spheres.",
  ing=["D64", "NN", "ANCT", "TOL8", "SOFT"])
e(id="S3", ph="p4", date="11 Nov", title="+ surface-following push", verdict="failed",
  idea="Variety and anchor pushes also act at each point's surface guess.", out="Same grow-then-collapse; gradients of 800–900.",
  ing=["D64", "NN", "ANCT", "TOL8", "SOFT"])
e(id="D1", ph="p4", date="11 Nov", title="Consistency dial 0.3", verdict="failed",
  idea="No filled centre; consistency's hold on the surface cut to 30%.", out="Same pattern, then out of GPU memory.",
  ing=["D64", "NN", "ANCT", "THIN", "TOL8", "SOFT", "NOC"])
e(id="E1", ph="p4", date="12 Nov", title="Eikonal only", verdict="mixed",
  idea="Consistency switched off entirely.", out="Variance 0.50 and 21 directions, all clean, but a rougher field and about 100 thin-wall sites per pass.",
  m=dict(var=0.502, dirs=21, clean=128, anch=0.81), ing=["D64", "NN", "ANCT", "TOL8", "SOFT"])
e(id="S4", ph="p4", date="12 Nov", title="Capped-follow", verdict="helped",
  idea="Surface-following push and fixed-surface consistency, with consistency's weight capped at 20.",
  out="Variance 0.76 against G2's 0.165 at 1,000 updates, 21 directions, 127/128 clean. Cost: field error four times its tolerance. Base of every later recipe.",
  m=dict(var=0.762, dirs=21, clean=127, anch=0.82),
  imgs=[("data_free_random_shapes_S4_follow_capped_1000", "S4 random codes"), ("data_free_field_error_capped_follow", "Field error, capped-follow"),
        ("data_free_field_error_ordinary", "Field error, ordinary (G2)")], ing=["D64", "NN", "ANCT", "TOL8", "CAP"])
e(id="S5b", ph="p4", date="12 Nov", title="S4 to 2,500", verdict="mixed",
  idea="S4 continued.", out="Variance 1.10, but shapes grew a third past the size target, which explains the gain. Directions 49–64 all decode to one neutral shape.",
  m=dict(var=1.104, dirs=22, clean=127, anch=0.86),
  imgs=[("data_free_random_shapes_S5b_2500", "S5b random codes"), ("data_free_direction_loops_S5b_2500_page1", "One loop per code direction")],
  ing=["D64", "NN", "ANCT", "TOL8", "CAP"])
e(id="T", ph="p4", date="12 Nov", title="Tidy-up fine-tunes (T1–T3)", verdict="failed",
  idea="Short fine-tunes of S4 that repair the field while holding the shapes.",
  out="Either the field was repaired by flattening the shapes (T1), or the shapes held and the field did not improve (T2, T3).")
e(id="R1", ph="p4", date="12 Nov", title="Field-weight ramp", verdict="nogain",
  idea="Eikonal and consistency weights ramped in over 500 updates.", out="Same as G2; early 'variety' was field error.",
  m=dict(var=0.123, dirs=14, clean=128, anch=0.84), ing=["D64", "NN", "ANCT", "TOL8", "RAMP"])

# ---------------- phase 5
e(id="H2", ph="p5", date="12 Nov", title="No filled centre", verdict="helped",
  idea="S4 recipe without the rule that the centre is solid; third anchor set with thin rods, plates, hooks.",
  out="128/128 clean, variance 1.24 (part of it from growth), flatter and thinner random shapes, thin shapes reachable.",
  m=dict(var=1.235, dirs=21, clean=128, anch=0.75), imgs=[("data_free_random_shapes_H2_2500", "H2 random codes")],
  ing=["D64", "NN", "ANCT", "THIN", "TOL8", "CAP", "NOC"])
e(id="H3", ph="p5", date="12 Nov", title="H2 to 7,000, size held", verdict="mixed",
  idea="H2 continued with the size-weight ceiling at 500.",
  out="Variance 1.45 at the target size, anchors 0.84, but directions fell to 17 and thin walls rose. Unseen fit 0.765, no better than H2.",
  m=dict(var=1.449, dirs=17, clean=126, anch=0.84, fit=0.765),
  imgs=[("data_free_random_shapes_H3_7000", "H3 random codes"), ("data_free_fit_six_unseen_H3_7000", "Unseen objects"),
        ("data_free_redundancy_paths_H3_7000", "Many codes give one shape")],
  ing=["D64", "NN", "ANCT", "THIN", "TOL8", "CAP", "NOC", "LONG"])
e(id="P12", ph="p5", date="12 Nov", title="Extent pose rule (P1, P2, X2)", verdict="failed",
  idea="Size measured by reach (6th power) instead of mass, with procedural anchors by kind of geometry.",
  out="Up to half the random codes decoded to an empty shape. X1 (same anchors, mass rule) was clean, so the pose rule was the cause.",
  imgs=[("data_free_procedural_anchors_extent6_v3", "Procedural anchors v3"), ("data_free_random_shapes_P1_1000", "P1: many empty or tiny shapes")],
  ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EXT"])
e(id="P3", ph="p5", date="12 Nov", title="Thin procedural anchors", verdict="failed",
  idea="Mass rule with anchors mostly under 0.16 across.", out="Clean to 1,250, then rods became strings of beads and shapes emptied.",
  imgs=[("data_free_random_shapes_P3_1500", "P3 at 1,500")], ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC"])
e(id="P4", ph="p5", date="12 Nov", title="Thick procedural anchors", verdict="mixed",
  idea="64 thicker procedural anchors in 47 families.",
  out="Valid and about equal to H2, anchors 0.82. The anchors' geometry stays near their codes: random shapes are still blob-like.",
  m=dict(var=1.231, dirs=20, clean=126, anch=0.82),
  imgs=[("data_free_procedural_anchors_thick_v5", "Thick procedural anchors"), ("data_free_random_shapes_P4_2500", "P4 random codes")],
  ing=["D64", "NN", "ANCP", "TOL8", "CAP", "NOC"])

# ---------------- phase 6
e(id="I1", ph="p6", date="12 Nov", title="Even-code penalty", verdict="helped",
  idea="Every code direction must change the decoder's inputs equally.",
  out="32 directions against 21; less total variance (0.50 against 0.76); validity unchanged.",
  m=dict(var=0.501, dirs=32, clean=127, anch=0.82), imgs=[("data_free_random_shapes_I1_1000", "I1 random codes")],
  ing=["D64", "NN", "ANCT", "TOL8", "CAP", "EVEN"])
e(id="I2", ph="p6", date="12 Nov", title="Even-code to 5,000", verdict="mixed",
  idea="I1 continued, size ceiling 500.", out="33–36 directions held all run, no wiggles; variance flat after 2,500; final checkpoint had 8 bad codes.",
  m=dict(var=0.874, dirs=36, clean=120, anch=0.88),
  imgs=[("data_free_random_shapes_I2_5000", "I2 random codes"), ("data_free_loop_waves_I2_vs_S5b", "No waves along code loops"),
        ("data_free_code_path_gains_I2_5000", "Where direction strength is lost")],
  ing=["D64", "NN", "ANCT", "TOL8", "CAP", "EVEN", "LONG"])
e(id="dead", ph="p6", date="13 Nov", title="Dead-unit finding", verdict="diag",
  idea="Unit activity measured in each layer.",
  out="Most of the last hidden layer dies: 99 units at 1,000 updates, 405 at 7,000. 96–99% of the variety gradient lands in the final layer.")
e(id="A1b", ph="p6", date="13 Nov", title="Leaky activation", verdict="helped",
  idea="Activation keeps a 0.05 slope when off.", out="9 dead units against 99, variance 0.645 against 0.50, nothing worse.",
  m=dict(var=0.645, dirs=33, clean=128, anch=0.82), ing=["D64", "NN", "ANCT", "TOL8", "CAP", "EVEN", "LEAK"])
e(id="N", ph="p6", date="13 Nov", title="Layer norm (N1, N2c)", verdict="nogain",
  idea="Layer norm on the last layer, or on all 8.", out="Did not stop unit death (88 and 142 dead). N2 at full points ran out of memory.",
  m=dict(var=0.599, dirs=32, clean=128, anch=0.81), ing=["D64", "NN", "ANCT", "TOL8", "CAP", "EVEN", "LN"])
e(id="K", ph="p6", date="13 Nov", title="Capacity: anchors only (K1, K2)", verdict="diag",
  idea="Decoder trained only to reproduce the 64 anchors; K2 with activation sharpness 100.",
  out="Levels off at 0.84 (0.86 with β 100). Thin rods lost, gear teeth and spikes rounded. The smooth network is part of the limit.",
  m=dict(anch=0.84),
  imgs=[("data_free_anchor_fit_K1_anchors_only_1500", "K1: anchors only"), ("data_free_anchor_fit_K2_anchors_only_beta100_1500", "K2: β 100")])
e(id="Fo", ph="p6", date="13 Nov", title="Fourier bands (anchors only)", verdict="mixed",
  idea="Sines and cosines of the coordinates: 3 bands (F1) or 5 (F2).",
  out="3 bands: crisper chunky objects, thin ones slightly worse, unit death tripled. 5 bands: ripples and specks.",
  m=dict(anch=0.84),
  imgs=[("data_free_anchor_fit_F1_anchors_only_fourier3_1500", "3 Fourier bands"), ("data_free_anchor_fit_F2_anchors_only_fourier5_1500", "5 bands: ripple and specks")])
e(id="J1", ph="p6", date="13 Nov", title="Procedural + even-code + no centre", verdict="helped",
  idea="The three newest ingredients together (stopped at 1,195).", out="At 1,000: variance 0.725, 29 directions, anchors 0.78.",
  m=dict(var=0.725, dirs=29, clean=126, anch=0.78), ing=["D64", "NN", "ANCP", "TOL8", "CAP", "NOC", "EVEN"])
e(id="J2", ph="p6", date="13 Nov", title="J1 + leak, 5,000", verdict="helped",
  idea="J1 with the leaky activation, full length.",
  out="128/128 clean, 29 directions; unseen fit 0.787, best at the time. Variety flat after 2,500.",
  m=dict(var=0.786, dirs=29, clean=128, anch=0.82, fit=0.787), imgs=[("data_free_random_shapes_J2_5000", "J2 random codes")],
  ing=["D64", "NN", "ANCP", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "LONG"])

# ---------------- phase 7
e(id="J4", ph="p7", date="13 Nov", title="J4: current base recipe", verdict="helped", best=True,
  idea="J2 plus 96 anchors (thick, thin rods, thin sheets), activation sharpness 100 and 3 Fourier bands.",
  out="Variety still growing at 5,000 (1.12), 31 directions, 128/128 clean. Unseen fit 0.809, best overall. Thin rods fit loosely (0.63).",
  m=dict(var=1.118, dirs=31, clean=128, anch=0.81, fit=0.809),
  imgs=[("data_free_random_shapes_J4_5000", "J4 random codes, update 5,000"), ("data_free_procedural_anchors_thick_plus_thin_v6", "The 96 anchors"),
        ("data_free_anchor_fit_thin_J4_5000", "The 32 thin anchors"), ("data_free_fit_six_unseen_J4_5000", "Six unseen objects"),
        ("data_free_worst_unseen_fits_J4_5000", "Worst unseen fits")],
  ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR", "LONG"])
e(id="J5", ph="p7", date="13 Nov", title="No thicken, field on anchors", verdict="hurt",
  idea="J4 without the thicken term, with field terms on anchors and a third of the anchor practice.",
  out="Worse everywhere: 19 bad codes, thin rods mostly in pieces, anchors 0.67.",
  m=dict(var=1.023, dirs=28, clean=109, anch=0.67), imgs=[("data_free_random_shapes_J5_5000", "J5 random codes")],
  ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR", "LONG", "NOTHK"])
e(id="V1", ph="p7", date="13 Nov", title="Surface-band variety (short)", verdict="mixed",
  idea="Variety compares shapes only near their surfaces (A1b recipe, 1,000 updates).", out="Same variety as A1b with half the thin walls; field no better.",
  m=dict(var=0.647, dirs=33, clean=128, anch=0.81), imgs=[("data_free_random_shapes_V1_1000", "V1 random codes")],
  ing=["D64", "NN", "ANCT", "TOL8", "CAP", "EVEN", "LEAK", "BAND"])
e(id="J6", ph="p7", date="13 Nov", title="J4 to 10,000", verdict="nogain",
  idea="J4 continued unchanged.", out="Wandered instead of improving; 17 bad codes at the end. The memory fault was active in its last hour.",
  m=dict(var=1.336, dirs=31, clean=111, anch=0.82), ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR", "LONG"])
e(id="J7", ph="p7", date="14 Nov", title="J6 + settle to 12,000", verdict="helped",
  idea="J6 continued at one-tenth learning rate.",
  out="Best anchors (0.88 at 11,000: thick 0.93, rods 0.76, sheets 0.82), validity mostly back. Unseen fit 0.815, barely above J4.",
  m=dict(var=1.27, dirs=31, clean=126, anch=0.88, fit=0.815), imgs=[("data_free_anchor_neighbourhood_J7_11000", "How far an anchor's shape reaches")],
  ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR", "LONG", "SETTLE"])
e(id="TA1", ph="p7", date="14 Nov", title="Trainable anchor codes", verdict="nogain",
  idea="J4 with anchor codes trainable after 500 updates, then a settle stage.",
  out="Same as J4. The settle stage raised anchors 0.79 → 0.88; unseen fit 0.809.",
  m=dict(var=1.024, dirs=31, clean=127, anch=0.88, fit=0.809), imgs=[("data_free_random_shapes_TA1s_6000", "TA1s random codes")],
  ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR", "LONG", "TRAIN", "SETTLE"])
e(id="TA2", ph="p7", date="14 Nov", title="Trainable codes + surface band", verdict="mixed",
  idea="TA1 with the surface-band variety measure.", out="Thin walls and gaps cut by a third to a half, variety a quarter lower, chunkier shapes; unseen fit 0.806.",
  m=dict(var=0.758, dirs=31, clean=126, anch=0.86, fit=0.806), imgs=[("data_free_random_shapes_TA2s_6000", "TA2s random codes")],
  ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR", "LONG", "TRAIN", "SETTLE", "BAND"])
e(id="C1", ph="p7", date="14 Nov", title="Crease term, weight 10", verdict="hurt",
  idea="Charge for curved surface, so flat faces and sharp edges are cheap.", out="Every random shape became an axis-aligned box; variance down a third, 9 directions.",
  m=dict(var=0.32, dirs=9, clean=127, anch=0.40), imgs=[("data_free_random_shapes_C1_1000", "C1: boxes")],
  ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR", "CREASE"])
e(id="C2", ph="p7", date="14 Nov", title="Crease 10 on mature J7", verdict="hurt",
  idea="Crease term added from J7's 11,000 checkpoint.", out="Limbs and detail erased; variance fell nine-fold (1.07 → 0.12).",
  m=dict(var=0.12, dirs=13, clean=127, anch=0.85), imgs=[("data_free_crease_C2_vs_J7_12000", "J7 vs C2, same codes")],
  ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR", "LONG", "SETTLE", "CREASE"])
e(id="C3", ph="p7", date="14 Nov", title="Crease term, weight 2", verdict="hurt",
  idea="C1 at weight 2.", out="No sharpening, and 44 of 128 codes had a through-hole. No middle ground between 2 and 10.",
  m=dict(var=0.51, dirs=20, clean=84, anch=0.54), imgs=[("data_free_crease_weights_0_2_10_at_1000", "Crease weight 0, 2 and 10 on the same codes")],
  ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR", "CREASE"])
e(id="W1", ph="p7", date="14 Nov", title="Weight norm", verdict="hurt",
  idea="J4 recipe, 1,000 updates, each unit's weights stored as direction and size, so the output layer stops taking the whole gradient.",
  out="Worse than J4 on every measure: 57/128 clean (most in pieces), anchors 0.25, 12 directions. The output layer's size fell 60-fold to where one step is as big as the number itself, so every shape's departure from the sphere swung between updates; hidden-unit sizes froze.",
  m=dict(var=0.53, dirs=12, clean=57, anch=0.25),
  imgs=[("data_free_random_shapes_W1_1000", "W1 random codes, many in pieces"), ("data_free_anchor_fit_W1_1000", "W1 anchors")],
  ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR", "WN"])
e(id="L1o", ph="p7", date="14 Nov", title="Final layer at 0.1× learning rate", verdict="pending",
  idea="J4 continued from 5,000 to 10,000 with the output layer's learning rate cut to a tenth, plus a running weight average. The output layer's tiny weights were changing 10–30× faster than any other layer, shifting all shapes at once.",
  out="Paused at 8,500 on 14 Nov and not evaluated. Training checks hover at clean 0.88–1.00 (one dip to 0.19 at 8,000), anchors about 0.81, near J6's. The decay variant (L2) was cancelled.",
  ing=["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR", "LONG", "OUTLR"])

# ---------------- phase 9: no fixed sphere
J4ING = ["D64", "NN", "ANCP", "THIN", "TOL8", "CAP", "NOC", "EVEN", "LEAK", "SHARP", "FOUR"]
e(id="S0", ph="p9", date="14 Nov", title="Can the network draw a sphere?", verdict="diag",
  idea="Fit the plain main network to a radius-0.5 sphere, as a starting point for a decoder with no fixed sphere base.",
  out="Yes: radius within 0.0003 in under three minutes, with or without Fourier bands, and all 32 random codes clean at every starting spread.",
  imgs=[("data_free_sphere_fit_fourier3", "Fitted sphere, 3 Fourier bands")])
e(id="S1", ph="p9", date="14 Nov", title="No base, fitted sphere start", verdict="hurt",
  idea="J4 recipe with the network's own output as the field (no fixed sphere), started from the fitted sphere with a small code spread.",
  out="All 128 clean, but shapes barely left the sphere: variance 0.14 against J4's 0.50, 10 directions, anchors 0.37, field error twice J4's.",
  m=dict(var=0.14, dirs=10, clean=128, anch=0.37),
  imgs=[("data_free_sphere_start_S1_vs_J4_1000", "S1 vs J4, same codes"), ("data_free_field_sphere_start_S1_update25", "Field after 25 updates")],
  ing=J4ING + ["NOBASE"])
e(id="S2", ph="p9", date="14 Nov", title="No base, third learning rate", verdict="nogain",
  idea="S1 at one third of the learning rate.", out="Almost no change; the field error sat at twice J4's with the slope weight at its ceiling.",
  m=dict(var=0.11, dirs=9, clean=128, anch=0.34), imgs=[("data_free_random_shapes_S2_1000", "S2 random codes")], ing=J4ING + ["NOBASE"])
e(id="S3", ph="p9", date="14 Nov", title="No base, usual spread + anchor ramp", verdict="hurt",
  idea="S1 with the usual code spread and the anchor pull ramped in.",
  out="Shapes broke up, shrank and drifted: 5/128 clean, anchors 0.10. The variety push broke the field by update 25.",
  m=dict(var=0.08, dirs=11, clean=5, anch=0.10), imgs=[("data_free_random_shapes_S3_1000", "S3: broken shapes")], ing=J4ING + ["NOBASE"])
e(id="Sfield", ph="p9", date="14 Nov", title="Field contours: pinned far field", verdict="diag",
  idea="Contour charts of the field in the no-base runs.",
  out="Found a flaw: blending the output into the sphere's field between radius 0.7 and 0.9 forced the far field to describe a sphere, so the slope term could never be met. Fixed with a smooth minimum against the bounding ball.",
  imgs=[("data_free_field_fixed_base_J4_update1000", "J4 field, for comparison")])
e(id="S4", ph="p9", date="14 Nov", title="No base, outer limit by minimum", verdict="failed",
  idea="S1 with the far-field fix.", out="Far field repaired, but every shape became a ring with a through-hole: 31/128 clean. The anchor pull drags all codes toward the average anchor early on.",
  m=dict(var=0.05, dirs=7, clean=31, anch=0.29),
  imgs=[("data_free_random_shapes_S4_1000", "S4: rings"), ("data_free_field_sphere_start_S4_update1000", "S4 field")], ing=J4ING + ["NOBASE"])
e(id="S5", ph="p9", date="14 Nov", title="No base that trains", verdict="mixed",
  idea="Small code spread, anchor pull off for 500 updates then ramped, slope term on near-surface points; 2,500 updates.",
  out="First no-base decoder that trains properly: field as accurate as J4's, limbs and dumbbells, still rising. Validity clearly worse (76/128).",
  m=dict(var=0.57, dirs=12, clean=76, anch=0.52), imgs=[("data_free_random_shapes_S5_2500", "S5 random codes")], ing=J4ING + ["NOBASE"])
e(id="S6", ph="p9", date="15 Nov", title="No base to 5,000", verdict="hurt",
  idea="S5 continued to 5,000.",
  out="Behind J4 on every model measure: 90/128 clean, 15 directions, anchors 0.64, unseen fit 0.768 (J4 0.809), worse even on thin objects. Shapes are thinner and more open, with fins and webs.",
  m=dict(var=0.77, dirs=15, clean=90, anch=0.64, fit=0.768),
  imgs=[("data_free_no_base_S6_vs_J4_5000", "S6 vs J4, same codes"), ("data_free_random_shapes_S6_5000", "S6 random codes"),
        ("data_free_layer_dims_no_base_vs_J4", "Code directions by layer: no base vs J4")], ing=J4ING + ["NOBASE", "LONG"])
e(id="Svar", ph="p9", date="15 Nov", title="Is the variance figure unfair to thin shapes?", verdict="diag",
  idea="Pairwise measures (overlap, surface distance, volume and area spread) on J4 and S6.",
  out="Partly. By overlap S6 is level with J4 and its shapes vary twice as much in volume and area, but surface distance to the nearest shape is 15% lower. The training's own field-difference measure rates thin shapes lowest.")
e(id="S7", ph="p9", date="15 Nov", title="No base to 10,000", verdict="nogain",
  idea="S6 continued to 10,000.",
  out="Variety and anchors kept rising (1.36, 0.71), validity and directions did not (85/128, 20). The first layer stayed at about 14 code directions. Unseen fit 0.763, no better.",
  m=dict(var=1.36, dirs=20, clean=85, anch=0.71, fit=0.763), imgs=[("data_free_random_shapes_S7_10000", "S7 random codes")], ing=J4ING + ["NOBASE", "LONG"])

# ---------------- phase 10: learned rounded-box base
e(id="B1", ph="p10", date="14 Nov", title="Learned box base, warm start", verdict="failed",
  idea="The fixed sphere replaced by a rounded box whose extents and rounding come from a learned head on the side network; warm start from J4 at 5,000.",
  out="Every code kept the same base: the head drifted into the flat part of its ramp within about 100 updates, so gradients could not reach it.",
  ing=J4ING + ["BOX", "LONG"])
e(id="B2", ph="p10", date="15 Nov", title="Fixed code-to-box map (cancelled)", verdict="diag",
  idea="Base set by fixed random code directions instead of a learned head.",
  out="Cancelled before it ran: a hand-wired map bypasses what the latent space is for. The base must come from a learned head.")
e(id="B3", ph="p10", date="15 Nov", title="Linear head + deviation budget, warm start", verdict="failed",
  idea="Head outputs the deviation from the sphere directly (no squash), with an adaptive budget on how far it may go.",
  out="The base did not move (base share of field variance 0.00). From a warm start the correction already makes every shape, so moving the base gains nothing.",
  ing=J4ING + ["BOX", "LONG"])
e(id="B456", ph="p10", date="15 Nov", title="Base head from scratch (B4, B5, B6)", verdict="helped",
  idea="Three head designs trained from scratch for 1,000 updates: linear on side features (B5), own network on the code (B4), side features → 64 tanh → 4 (B6).",
  out="From scratch the base is used. B6 was the healthiest (clean 1.00) and the most varied by every pair measure, 12% above J4 at 1,000; B4 and B5 made shapes fatter.",
  imgs=[("data_free_box_base_B6_1000", "B6 bases and shapes"), ("data_free_box_base_B5_1000", "B5"), ("data_free_box_base_B4_1000", "B4")],
  ing=J4ING + ["BOX"])
e(id="B7", ph="p10", date="15 Nov", title="B6 to 5,000", verdict="nogain",
  idea="B6 continued with a wider deviation budget.",
  out="The base faded back to one shared rounded cube (base share 0.16 → 0.03). Same variety as J4, slightly thicker and healthier (clean 1.00, anchors 0.83).",
  imgs=[("data_free_box_base_B7_5000", "B7: bases converge")], ing=J4ING + ["BOX", "LONG"])
e(id="B89", ph="p10", date="15 Nov", title="Keeping the base in use (B8, B9)", verdict="mixed",
  idea="From B6 at 1,000 to 3,000: field gradients into the base scaled 0.1× (B8), or a push to spread the bases apart (B9).",
  out="B8 only slowed the fade. B9 kept rods, plates and boxes as bases and doubled the spread in size and area, but nearest neighbours were no further apart, and validity and anchors fell.",
  imgs=[("data_free_box_base_B9_3000", "B9: varied bases")], ing=J4ING + ["BOX", "LONG"])
e(id="B10", ph="p10", date="16 Nov", title="Slower base head", verdict="mixed",
  idea="B9 continued to 5,000 with the base head at 0.1× learning rate.",
  out="Health recovered and the base keeps 80% of the variety, but the decoded shapes are no more varied: the correction pads thin plate and bar bases out into blobs.",
  imgs=[("data_free_box_base_B10_5000", "B10: thin bases padded into blobs")], ing=J4ING + ["BOX", "LONG"])
e(id="B11", ph="p10", date="16 Nov", title="Cost on correction size (25)", verdict="mixed",
  idea="B10 continued to 7,000 with a weak cost on the correction's size near the surface.",
  out="Small gains (clean at every check from 5,750, nearest-shape distance up slightly), but the correction shrank only 14% and still pads outward. No control yet.",
  imgs=[("data_free_box_base_B11_7000", "B11 bases and shapes")], ing=J4ING + ["BOX", "LONG", "CORR"])
e(id="B12", ph="p10", date="16 Nov", title="Correction cost 50", verdict="pending",
  idea="B11 continued to 9,000 with the correction cost doubled.",
  out="Finished; not yet evaluated. Training checks at 9,000: probes clean 1.00, 7.8 live directions, anchors 0.79.",
  imgs=[("data_free_box_base_B12_9000", "B12 bases and shapes")], ing=J4ING + ["BOX", "LONG", "CORR"])
e(id="B13", ph="p10", date="16 Nov", title="Correction cost 100", verdict="pending",
  idea="B12 continued to 11,000 with the cost doubled again.",
  out="Finished; not yet evaluated. Training checks at 11,000: probes clean 1.00, 10.5 live directions, anchors 0.78, fewer thin walls (101 per pass against 189).",
  ing=J4ING + ["BOX", "LONG", "CORR"])
e(id="B12c", ph="p10", date="16 Nov", title="Cost 25 control to 9,000", verdict="running",
  idea="B11 continued to 9,000 with the cost unchanged, the control for B12.",
  out="Training (update 8,537 of 9,000 at 19:16 on 16 Nov). Checks so far: clean 0.94–1.00, anchors 0.82.",
  ing=J4ING + ["BOX", "LONG", "CORR"])

# ---------------- phase 8
for i, (t, d, o) in enumerate([
    ("Sphere-window decoder", "Field forced to the sphere's distance beyond a radius.", "Replaced before training: not a true distance field, would pull shapes back to spheres."),
    ("Zero FiLM init", "Code heads start at zero so every code is identical.", "Variety gradient was exactly zero; replaced by a small random init."),
    ("Barrier over all samples", "Barrier averaged over all 2,048 points.", "Diluted 2,000×; now averaged near the surface."),
    ("Consistency tolerance schedule", "Loose tolerance early, tightened late.", "Implemented, never trained."),
    ("Gradient surgery (PCGrad)", "Project away conflicting gradients.", "Measured conflict (cosine −0.95, 98% in the last layer); not built, as it would remove nearly all of consistency."),
    ("Coverage term on statistics", "Each statistic group must span a target range.", "Proposed, never built."),
    ("Cosine code features", "Reviewer proposal for richer code input.", "Never run."),
    ("Local thickness push", "Per-point thickness from facing wall pairs, pushed apart.", "Proposed, never built."),
]):
    e(id=f"x{i}", ph="p8", date="", title=t, verdict="diag", idea=d, out=o)

POTENTIAL = [
    ("high", "J4 recipe at 32-D codes", ["D32", "EVEN", "NOC", "LEAK"],
     "Code size 32 is the stated target, but 32-D was last tried on the old recipe (L2/F-series). The even-code penalty, no filled centre, leak, β 100 and Fourier bands have only ever run at 64-D."),
    ("high", "Many more anchors (256–512 procedural)", [],
     "An anchor's shape fades halfway by about 50° from its code, and random codes sit at least 52° from every anchor. With 96 anchors the space between them stays blob-like. The anchor count was only varied 64 → 96."),
    ("high", "J4 seed repeat at 5,000", [],
     "Seed noise was measured only at 1,000 updates on the old recipe. Every recipe comparison since is single-seed, and differences under about 0.2 in variance read as noise."),
    ("med", "Wider decoder (width 1024)", [],
     "Width 512 was never varied. The anchors-only capacity test levels off at 0.84 and rounds detail, and most of the last layer dies, which both point at the network."),
    ("med", "Periodic settle stages", ["LONG", "SETTLE"],
     "Every settle stage helped (F2s, G3s, J7, TA1s), always once at the end. Alternating normal and one-tenth learning rate, or a cosine schedule, is untried."),
    ("med", "GINN-style concave aggregation and RMS-scaled multipliers", ["NN"],
     "Published work finds a concave (square-root) transform of nearest-neighbour distances spreads shapes evenly, and scales penalties by the constraint's running RMS. Several multipliers here sit at their caps."),
    ("med", "Spread term with even-code penalty and anchors", ["SPR", "EVEN"],
     "Spread raised effective directions from 2 to 7 in 32-D without anchors. It was never tried after the switch to nearest-neighbour variety, anchors or the even-code penalty."),
    ("med", "Finish the final-layer learning-rate run (L1)", ["OUTLR", "SETTLE"],
     "Paused at 8,500 with no evaluation. It targets the one mechanism measured behind J6's late swings: the tiny output layer moving every shape at once. A settle stage after it is also untried."),
    ("med", "Box base with anchor-box targets", ["BOX", "ANCP"],
     "The learned base fades unless something gives it a job. Fitting a box to each anchor and asking the head to match it at the anchor code is the other session's untried candidate, and keeps the base learned."),
    ("low", "Neighbour pool with anchors", ["NN", "ANCP"],
     "The 256-code pool run (s) crashed while promising at update 500 and was never re-run."),
    ("low", "Leak + layer norm", ["LEAK", "LN"],
     "Layer norm alone did not stop unit death; the leak did. Together they are untried, but there is little reason to expect a gain."),
    ("low", "Crease term late or ramped", ["CREASE", "SETTLE"],
     "Weight 2 did nothing and 10 erased variety. A small weight only near the end might sharpen edges, but both tests suggest it simplifies shapes first."),
    ("low", "Fourier bands without β 100", ["FOUR"],
     "J4 changed anchors, sharpness and Fourier bands at once, so each one's share of the gain is unknown."),
]
