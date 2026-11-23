# Data-free latent runs: notes for the run monitor

Maintained by the main Claude session (the one working in
`/home/eric-luo/Downloads/pointcloud-motion-planning/.claude/worktrees/data-free-connectivity-plan`).
It rewrites the "Now" section every time a run is started, queued, stopped or finishes.
Last edited: 2026-11-22 23:35 CST.

For live numbers always run the status script first; this file explains what they mean.

    python3 /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/monitor/status.py
    python3 /home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/monitor/status.py <run folder name>

## What the project is

A decoder-only 3-D shape generator: a network turns a code (a unit vector, 32 or 64 numbers) into a solid
shape, with no shape dataset and no encoder. Random codes should give valid, varied solids; these are later
used as obstacles for training a robot-arm motion planner. Training balances a push for variety between
shapes against constraints that keep each shape a valid single solid. Since 2026-11-11 a few real objects
("anchors", 64 solids from Thingi10K) are also reproduced at fixed codes, which stops the code's directions
from going unused.

## Now

**Queued 23:35 CST 22 Nov (rounded-box-base session, owner's request): `SH3c_SH3_cont_variety_controller_to10k`.** Starts
when the GPU has had no compute process for three checks a minute apart (the `dihfreq.rl` grasp job holds it now). SH3
continued 5,000 -> 10,000 with a variety weight that moves between 1 and 4: +0.1% per update while smoothed consistency
< 0.025 and bridge/delete/fill/widen are within tolerance, else -0.2%. Slope tolerance 0.010 -> 0.015, multiplier limit
200 -> 1000. Corners beyond radius 0.9 stay unsampled; the owner will fine-tune them at the end. Watch `variety/weight`,
`variety/safe`, thickness and anchor fit. Commit `52c2223`.

**Corrections, 22 Nov (rounded-box-base session):** (1) SH1c's cap of 32 bound on 83% of updates from 5,000 to 6,700;
there is no uncapped control, so 5,000 -> 6,700 gains cannot be credited to training alone. SH1's slope multiplier
swings widely (11-140), and 33.4 at 5,000 was one sample. (2) Logged eikonal/consistency are on each run's own points.
On identical points (64 codes, `seed_gap/same_points.py`), eikonal / consistency: uniform box J4 0.0099 / 0.0393, SH1
0.0198 / 0.0758, SH3 0.0218 / 0.0717; shells J4 0.0379 / 0.0354, SH1 0.0097 / 0.0212, SH3 0.0075 / 0.0173. The shell
runs are about 2x WORSE on uniform box points, which include the corners beyond radius 0.9 that shells never sample;
`sdf_truth.py` only tested inside 0.9, so "better at every distance" holds only there. Outside 0.9 the decoder
guarantees the sign, not the distance. (3) SH3 also changed 4x32 -> 8x16 passes; only speed was checked for that
split, so SH3 vs SH1 is not a pure width comparison. (4) The slope multiplier limit is 200, not 10,000.

**Finished 17:53 CST 21 Nov: `SH1_J4_scratch_shells2048_shared_volw_5k`** (J4 from scratch with 2,048 shared shell
points for every term; pose/volume integrals volume-weighted; branch `rounded-box-base`, commit `5c1c1c2`). Result at
5,000 vs J4 at 5,000 (96 fresh codes unless noted): less varied (nearest 1 - overlap 0.428 vs 0.475, nearest surface
distance 0.058 vs 0.069), smaller and thinner (fill 0.080 vs 0.098, half-thickness median 0.222 vs 0.244), less
concave (solidity 0.668 vs 0.607 on 64 codes), smoother blobs with fewer creases in the pictures; fresh-code topology
126/128 clean (2 handles) vs J4 128/128; anchors IoU 0.80 vs 0.80. The 16 training probe codes stayed clean all run
(J4 dipped to 0.81 at 1,500). The eikonal multiplier ran higher (99 vs 39 early, 45 vs 32 late): on shells the slope
constraint sees more near-surface points and is harder to hold, which plausibly smooths the shapes. Pictures:
`figures/data_free_random_shapes_SH1_5000.png`, `figures/data_free_anchor_fit_SH1_5000.png`.

**`SH3_SH1_recipe_width768_5k` finished 07:17 CST 22 Nov.** At 5,000 vs SH1 at 5,000 (same recipe, main width 768 vs 512):
variety about the same (nearest 1 - overlap 0.424 vs 0.428, nearest surface distance 0.062 vs 0.058; J4 0.475 / 0.069),
slightly less concave (solidity 0.695 vs 0.668), same thickness (half-thickness 0.242 vs 0.222); more accurate field
(slope error within 0.02 of the surface 0.080 vs 0.097, value error at 0.02-0.05 0.0026 vs 0.0032, far field 0.016 vs
0.020), lower eikonal multiplier late (39 vs 45), topology 128/128 (SH1 126/128), anchors IoU 0.87 (SH1 0.80). Faster
early: nearest 1 - overlap 0.370 at 1,000 (SH1 0.277). The extra width is used (middle layers carry about 1.5x the
directions: code-dependent effective rank 164-244 vs 125-165), but it went into field accuracy, not variety. Last
layer still 45% never-active. Picture `figures/data_free_random_shapes_SH3_5000.png`.

**Run notes for SH3:** SH1's
recipe (J4 + 2,048 shared shells; from scratch, seed 0, 5,000 updates) with the main network widened 512 -> 768 (side
network unchanged, about 7.3M parameters vs 4.0M). Each update is 8 passes of 16 codes instead of 4 of 32 (same 128
codes; needed for the 0.75 memory cap) and 3 anchors per pass instead of 6 (same share). Logged "div" values are twice
SH1's because each pass divides by 16 codes instead of 32 (the gradient is the same). About 3.5 s per update, about
4.7 h. Question: does a wider main network make the slope constraint cheaper to hold and leave room for variety?
Compare with SH1 and J4 at 1,000 / 2,500 / 5,000 on variety, concavity, topology, true-SDF accuracy
(`analysis_scripts/seed_gap/sdf_truth.py`), the eikonal multiplier, and per-layer rank / dead units. Branch
`rounded-box-base`, commit `48383fe`; launcher `queue_logs/launch_SH3_SH1_recipe_width768_5k.sh`.

**`SH1c_SH1_cont_eikcap16_from6700_release8k_to10k` finished 00:01 CST 22 Nov.** Result (96 fresh codes for variety, 64
concavity, 32 true-SDF accuracy, 128 topology): nearest 1 - overlap SH1 5,000 0.428 -> 6,700 0.441 (cap 32, binding 83% of updates) ->
8,000 0.462 (end of cap 16) -> 9,500 0.422 -> 10,000 0.405; nearest surface distance 0.058 -> 0.064 -> 0.064 -> 0.062 ->
0.060; solidity 0.668 -> 0.616 -> 0.597 -> (10,000) 0.671; half-thickness 0.222 -> 0.243 -> 0.224 -> 0.256 -> 0.269.
Slope error within 0.02 of the surface: SH1 0.097, 8,000 0.131, 10,000 0.101; value error at 0.02-0.05: 0.0032, 0.0040,
0.0029; far field (0.4+) 0.0195 -> 0.0140. Topology 128/128, anchors IoU 0.88. So the loose phase gained variety and
concavity (level with J4 on concavity at 8,000) but the final tightening gave them back (overlap variety below SH1 at
5,000, solidity back to SH1's), while restoring and slightly improving field accuracy. Picture
`figures/data_free_random_shapes_SH1c_10000.png` (chunky, faceted shapes).

**Run notes for SH1c:** SH1 (J4
recipe on 2,048 shared shells) continued from 5,000 to 10,000 to loosen the slope (eikonal) constraint, let variety
grow, then restore accuracy. 21:12 -> 22:10 CST 21 Nov it ran 5,000 -> 6,700 with the eikonal multiplier capped at 32
(J4's late mean), and the cap DID bind (at 32 on 83% of updates), so that stretch was already mildly loosened, not plain SH1 training (snapshot `checkpoint_6700_cap32.pt`). Resumed 22:11 at 6,700 with the cap at 16 until 8,000, raised
geometrically to 100 over 8,000 -> 9,500, uncapped for the last 500 updates (the multiplier now sits at 16). Learning
rate is constant in this recipe. Compare with SH1 and J4 at 5,000 on variety, concavity, topology and true-SDF accuracy
(`analysis_scripts/seed_gap/sdf_truth.py`); snapshots at 6,000 / 6,700 / 8,000 / 9,500. Branch `rounded-box-base`, commit
`48383fe`; launchers `queue_logs/launch_SH1c_SH1_cont_eikcap32_release8k_to10k.sh` (first part),
`queue_logs/resume_SH1c_SH1_cont_eikcap16_from6700_release8k_to10k.sh`. (The folder was renamed from
`SH1c_SH1_cont_eikcap32_release8k_to10k`; its metrics.jsonl has steps 6,700-6,780 twice, the cap-16 rows last.)

**Distance-field accuracy of the shell runs (measured 21 Nov, `analysis_scripts/seed_gap/sdf_truth.py`: 32 random
codes, true signed distance to each model's own surface).** Shells make the field a much better distance field at
every distance, most near the surface. Slope error ||g| - 1| within 0.02 of the surface: J4 0.247 vs SH1 0.097; S6 0.254
vs SH2c 0.076. Value error |f - d| at 0.02-0.05: J4 0.0078 vs SH1 0.0032; S6 0.0077 vs SH2c 0.0025 (relative 22% vs 9%,
22% vs 7%). Far field (0.4+): 0.028 vs 0.020; 0.034 vs 0.019. So the shell runs trade some variety and concavity for a
2.5-3x more accurate field near the surface.

**`SH2c_SH2_continued_to_5k` finished 20:42 CST 21 Nov** (SH2 continued to 5,000 as S6 continued S5). At 5,000 vs S6:
fresh-code topology 128/128 clean vs 90/128 (S6: 35 handles); variety recovered from SH2's collapse (nearest 1 - overlap
0.375 vs 0.297 at 2,500) and the shared template is gone, but still behind S6 (0.375 vs 0.465; nearest surface distance
0.056 vs 0.058); thicker (half-thickness 0.248 vs 0.205) and less concave (solidity 0.729 vs 0.609). J4 at 5,000 (fixed
sphere, uniform points): 0.475 / 0.069, 128/128, solidity 0.607, so the sphere residual still gives more varied
shapes. Picture `figures/data_free_random_shapes_SH2c_5000.png`.

**`SH2_S5_recipe_no_sphere_base_shells2048_2500` finished 19:18 CST 21 Nov.** SH2 = the other session's S5
recipe (no fixed sphere, fitted-sphere start) with only the sample points changed to shells. Result at 2,500 vs S5 at
2,500: fresh-code topology 126/128 clean vs 75/128 (S5: 47 handles, 6 in pieces), but variety collapsed: nearest
1 - overlap 0.297 vs 0.371, nearest surface distance 0.043 vs 0.049, thicker (half-thickness 0.272 vs 0.196), less
concave (solidity 0.746 vs 0.686), and in the pictures almost every shape is the same template (rounded body with
one small knob at the lower left). So the clean topology mostly comes from shapes that barely vary. Anchors IoU 0.59
vs 0.52 (training diagnostic). Pictures `figures/data_free_random_shapes_SH2_2500.png` (S5's:
`figures/data_free_random_shapes_S5_2500.png`).

Original plan: **Training since 17:54 CST 21 Nov (queued after SH1; owner's request):
`SH2_S5_recipe_no_sphere_base_shells2048_2500`, then `SH2c_SH2_continued_to_5k`.** The other session's S5 recipe
(J4 recipe without the fixed sphere in the formula, started from the fitted sphere S0, outer limit by smooth minimum,
anchor pull held off 500 then ramped over 500, slope term at all points; seed 0, 2,500 updates) with only the sample
points changed as in SH1. Compare with S5 at 500 / 1,000 / 2,000 / 2,500 (S5's clean share was 0.44 / 0.38 at
1,000 / 2,500: many codes in pieces or with a through-hole). SH2c resumes SH2 to 5,000 exactly as S6 continued S5
(cosine schedule restarts over 5,000); compare with S6 at 3,000 / 4,000 / 5,000 (S6 clean 0.62 at 5,000). Question:
do the shells improve topology and stability enough to drop the sphere residual? Launcher
`queue_logs/queue_SH2_S5_recipe_no_sphere_base_shells2048_2500_then_continue.sh`.

**Paused 19:12 CST 14 Nov at the owner's request: `L1_J4_5000_final_layer_lr_tenth_to_10k` at update 8,500
(checkpoint saved; snapshots at 6,000 and 7,500 kept). `L2_J4_5000_final_layer_lr_tenth_decay_to_10k`
cancelled before it started (no folder). Resume L1 with `tmp/resume_run.sh L1_J4_5000_final_layer_lr_tenth_to_10k
10000` (its settings are in the checkpoint).**

**Nothing of this session is training or running (as of 03:15 CST 18 Nov). The machine hung and rebooted at
about 00:10 CST 18 Nov; S10 was resumed from its checkpoint at 18,700 and finished.**

**S10 finished 01:07 CST 18 Nov (owner's request: S9 continued from 12,000 to 20,000 at the low learning rate,
to see whether the limits of the sphere-residual start keep fading): it reached J7's fit to real objects, kept a
more accurate field at the surface with little dependence on which way the surface faces, and is more varied;
validity is lower and swings.**

| | S9 12,000 | S10 14,000 | 16,000 | 18,000 | 20,000 | J7 11,000 / 12,000 |
|---|---|---|---|---|---|---|
| Unseen-object overlap / surface distance | 0.808 / 0.0216 | - | 0.809 / 0.0214 | - | 0.816 / 0.0206 | 0.815 / 0.0207 |
| Fresh clean / 128 | 117 | 115 | 99 | 105 | 118 | 126 / 122 |
| Anchors | 0.86 | 0.87 | 0.87 | 0.87 | 0.88 | 0.88 / 0.84 |
| Total variance, 96 codes | 1.60 | 1.67 | 1.60 | 1.51 | 1.42 | 1.27 / 1.07 |
| Solid directions | 27 | 27 | 27 | 25 | 25 | 31 / 26 |
| Slope error within 0.03 of the surface | 0.25 | 0.24 | 0.30 | 0.25 | 0.19 | 0.34 / 0.49 |
| Slope error outward / past side-on | 0.20 / 0.31 | 0.21 / 0.29 | 0.24 / 0.40 | 0.21 / 0.33 | 0.17 / 0.28 | 0.28 / 0.63 (11k); 0.50 / 0.73 (12k) |
| Surface past side-on | 5.4% | 7.3% | 10.5% | 8.7% | 4.7% | 1.5% / 2.4% |

- Paired, surface distance: S10 at 20,000 minus S9 -0.0010 +/- 0.0002 (better on 75% of objects); minus J7 at
  11,000 -0.0000 +/- 0.0003 (level; overlap +0.002 +/- 0.004). All 225 attempts in one piece.
- Pairwise variety at 20,000: 1 - overlap to the nearest shape 0.522 (J7 0.508), surface distance 0.060 (0.066);
  half-thickness median 0.177 (0.236). Layers 3 / 4 / 5 still carry 15 / 56 / 22% of the surface response.
- One anchor code (76, a thin rod) developed a narrow spike in its field: minimum -1.8 at 18,000, -779 at 20,000,
  at radius 0.28 (not there at 12,000 or 16,000; weights finite; not crash damage, it was already growing before
  the crash). Random codes are unaffected (largest |field| about 1.5, at the box corners). Anchors are exempt from
  the slope and consistency terms, and without the sphere base the output has no bound, so nothing stops a
  spike in a small region the anchor pull rarely samples.
- No fixed-base run has been trained to 20,000 at this rate, so how much of the gain is just more training is
  not known.
- Pictures `figures/data_free_random_shapes_S10_20000.png`, `figures/data_free_anchor_fit_S10_20000.png`.

**S9 finished 19:05 CST 17 Nov (owner's request: S8 continued from 10,000 to 12,000 at a tenth of the learning
rate, as J7 did for J6): the lower rate recovered the fit to real objects; the base-free model is now level with
J4 at 5,000 and close to J7.**

| | S9 at 12,000 (no base) | J7 at 11,000 (base) | J7 at 12,000 |
|---|---|---|---|
| Unseen-object overlap / surface distance | 0.808 / 0.0216 | 0.815 / 0.0207 | not fitted |
| Fresh clean / 128 | 117 (121 / 123 / 117 at 10,500 / 11,000 / 11,500) | 126 | 122 |
| Anchors (worst) | 0.86 (0.51) | 0.88 | 0.84 |
| Total variance, 96 codes | 1.60 +/- 0.11 | 1.27 +/- 0.06 | 1.07 +/- 0.05 |
| Solid directions | 27 | 31 | 26 |
| 1 - overlap / surface distance to the nearest shape | 0.555 / 0.062 | 0.508 / 0.066 | - |
| Half-thickness median, fill | 0.171, 0.049 | 0.236, 0.080 | - |
| Slope error near the surface, random codes | 0.205 | 0.256 | - |

- Paired on the same 225 objects (surface distance): S9 minus S8 -0.0021 +/- 0.0002 (better on 87%); S9 minus
  J4 at 5,000 +0.0003 +/- 0.0003; S9 minus J7 at 11,000 +0.0010 +/- 0.0003 (J7 better on 72%).
- Layer use unchanged: layers 3 / 4 / 5 carry 23% / 49% / 17% of the surface response (J7 at 12,000: 18 / 54 / 18).
- Charts `figures/data_free_unseen_fit_chart.png` and `figures/data_free_unseen_fit_shapes.png` (all seven
  models); 64 codes in `figures/data_free_random_shapes_S9_12000.png`.

**J6 at 10,000, unseen-object fit (S8's proper control):** overlap 0.799, surface distance 0.0226, F at 0.02 /
0.04 0.60 / 0.85, thinnest third 0.710 / 0.0273, 222 of 225 attempts in one piece. Paired against it on the
same objects, S8 at 10,000 is worse by 0.0011 +/- 0.0004 in surface distance, 0.008 +/- 0.004 in overlap and
0.034 +/- 0.010 in F at 0.02 (`analysis_scripts/paired_fit.py`): a small but real gap.

**Distilled copy at 5,000 (D0), unseen-object fit:** overlap 0.806, surface distance 0.0218 (J4 at 5,000: 0.809 /
0.0213; paired difference +0.0004 +/- 0.0002), so distillation itself cost almost nothing. From 5,000 to 10,000
at the constant learning rate both models got worse at reaching unseen objects: J4 to J6 +0.0013 +/- 0.0004 in
surface distance, the distilled copy to S8 +0.0020 +/- 0.0003. The no-base model loses about 50% more, which
is S8's whole gap to J6.

**S8 finished 15:17 CST 17 Nov (owner's idea): the base-free decoder started as a copy of J4 at 5,000 keeps
J4's health and its use of the code, and has a more accurate field at the surface than J4 continued with the
base (J6).** `D0_distill_J4_5000_no_base`: the base-free decoder with J4's weights, trained for 5,000 updates
to reproduce J4's field (`scripts/distill_data_free.py`, commit `6e3965e`): field error 0.21 to 0.009, sign
disagreement 30% to 0.7%, overlap with J4's shapes 0.10 to 0.93 (worst 0.89), 128 / 128 clean, anchors 0.81.
`S8_distilled_no_base_continued_to_10k`: that copy trained on from 5,000 to 10,000 with the J4 recipe's
losses and no base. Control: J6.

| | S8 (no base) | J6 (J4 continued, base kept) |
|---|---|---|
| Fresh clean / 128 at 6,000 / 7,000-7,500 / 9,000 / 10,000 | 126 / 123 / 128 / 120 | 126 / 128 / 127 / 111 |
| Anchors at the same points | 0.71 / 0.72 / 0.72 / 0.74 | 0.76 / 0.80 / 0.73 / 0.82 |
| Total variance, 96 codes, same points | 0.82 / 1.08 / 0.97 / 1.32 | 1.09 / 1.06 / 0.84 / 1.42 |
| Solid directions at 10,000 | 27 | 32 |
| Layer shares of the surface response (layers 3 / 4 / 5) at 10,000 | 23% / 43% / 23% | 23% / 45% / 21% |
| Surface directions above 1% at 10,000 | 50 | 56 |
| 1 - overlap / surface distance to the nearest shape at 10,000 | 0.533 / 0.061 | 0.583 / 0.063 |
| Half-thickness median at 10,000 | 0.148 | 0.197 |
| Surface near side-on / past side-on at 10,000 | 39% / 4.5% | 42% / 4.8% |
| Slope error within 0.03 of the surface at 10,000 (outward / side-on / past) | 0.21 (0.15 / 0.25 / 0.28) | 0.45 (0.44 / 0.48 / 0.60) |

- Unseen-object fit at 10,000 (225 objects): overlap 0.792, surface distance 0.0237, F at 0.02 / 0.04 0.56 /
  0.84, thinnest third 0.706 / 0.0278, all 225 attempts in one piece (S7, no base from a fitted sphere: 0.763 /
  0.0259; J4 at 5,000: 0.809 / 0.0213; J7 at 11,000: 0.815 / 0.0207). J6 at 10,000 and the distilled copy at
  5,000 are being measured.
- The distilled start kept J4's internal structure (every layer gets about 48 code directions; layers 3 to 5
  carry the shape), unlike the no-base runs started from a fitted sphere, and it stayed that way to 10,000.
- Picture `figures/data_free_distilled_no_base_vs_J4.png`: the copy is indistinguishable from J4 at 5,000; by
  10,000 shapes are thinner and more limb-like. 64 codes: `figures/data_free_random_shapes_S8_10000.png`.

**Nothing of this session is training or running (as of 23:55 CST 15 Nov; the memory fault was active again
this afternoon, 58 damaged cached files and several segfaults, and 7 more at 22:30). `L1_J4_5000_final_layer_lr_tenth_to_10k` is still paused at
update 8,500; L2 was cancelled before it started.**

**Where the code's directions go missing in the no-base decoder (owner's question, measured 17 Nov,
`analysis_scripts/layer_inject.py`, chart `figures/data_free_layer_inject_no_base_vs_J4.png`).** The code
reaches every main layer through that layer's scale and shift; each layer was moved alone.
- J4: every layer gets about 53 directions; control of the surface is spread over layers 3 to 5 (22% / 42% /
  19% at 5,000); all 63 directions reach the surface measurably.
- No base (S5 to S7): layers 1 and 2 get only 17 to 19 directions by 10,000 (layers 4 to 8 still 42 to 49), the
  later layers' code barely reaches the surface (under 5% together), and 84% of the surface's response comes
  through layer 2; 32 directions reach the surface measurably.
- Separating the causes with the other no-base runs (all no base and fitted start; S1, S4 and S5 with the small
  code spread 0.005, S3 with the usual 0.1): the narrowing of the code going into layers 1 and 2 appears in
  S1, S4 and S5 but not in S3 (42 to 46 directions at every layer), so it comes from the small starting spread.
  Control of the surface concentrated in the first layers appears in all of them, S3 included (layer 1: 39 to
  78% of the surface response at 500 to 1,000), and not in J4, so it comes from the missing base and/or the
  fitted start (not separable with these runs). Delayed anchors, the slope term at the surface and the outer
  limit are ruled out (S1 and S4 show the pattern without them).

**S7 finished 14:55 CST 15 Nov (no-base decoder continued from 5,000 to 10,000, owner's request): it keeps
improving in variety and anchor fit, not in validity or code directions.**

| | 5,000 (S6) | 6,000 | 7,000 | 8,000 | 9,000 | 10,000 | J6 at 10,000 |
|---|---|---|---|---|---|---|---|
| Fresh clean / 128 | 90 | 78 | 113 | 79 | 64 | 85 | 111 |
| Anchors | 0.64 | 0.65 | 0.67 | 0.77 | 0.73 | 0.71 | 0.82 |
| Total variance, 96 codes | 0.77 | 0.84 | 0.63 | 1.09 | 1.08 | 1.36 | 1.42 |
| Solid directions | 15 | 17 | 16 | 20 | 19 | 20 | 32 |
| Main layer 1, effective code directions | 13 | 13 | 14 | 14 | 14 | 14 | 49 |
| Surface directions above 1% (of 63) | 28 | 30 | 29 | 32 | 29 | 29 | 59 |

- Pairwise variety at 10,000 equals J6's (1 - overlap to the nearest shape 0.561 against 0.583, surface
  distance to the nearest 0.063 against 0.063, mean over pairs 0.123 against 0.114). Both models' shapes
  shrank and thinned by 10,000 (fill 0.058 / 0.052, half-thickness 0.169 / 0.197).
- The first main layer stayed at about 14 effective code directions throughout: the bottleneck did not open.
- Unseen-object fit at 10,000 (rerun from a clean grid file gave identical overlaps): overlap 0.763, surface
  distance 0.0259, F at 0.02 / 0.04 0.51 / 0.81, thinnest third 0.668 / 0.0303 (S6 at 5,000: 0.768 / 0.0264;
  J4 at 5,000: 0.809 / 0.0213; J7 at 11,000: 0.815 / 0.0207). No improvement from 5,000 to 10,000.
- Pictures `figures/data_free_random_shapes_S7_10000.png`, `figures/data_free_anchor_fit_S7_10000.png`.

**B11 finished 02:56 CST 16 Nov (other session's result, entry supplied by that session; B10 continued from
5,000 to 7,000 with a weak cost on the correction's size near the surface, weight 25): small gains, the cost is
too weak.** The correction near the surface shrank only 14% (mean square 0.0135 to 0.0116, still padding
outward +0.043). At 7,000 against B10 at 5,000 (and J4 at 5,000): 1 - overlap to the nearest shape 0.461 /
0.445 / 0.475 (+/- 0.006), surface distance to the nearest 0.0692 / 0.0690 / 0.0688, compactness median 0.34 /
0.39 / 0.34, half-thickness median 0.247 / 0.252 / 0.244, fill 0.110 / 0.120 / 0.098; clean 1.00 / 0.94 / 0.94
(clean at every check from 5,750), live 8.0 / 8.0 / 6.25, anchors 0.81 / 0.79 / 0.80; base variance share 0.59
(0.71 to 0.87 at earlier checks). No control without the cost from B10 yet, so part of the change may be the
extra training. Picture `figures/data_free_box_base_B11_7000.png`.

**B10 finished 00:12 CST 16 Nov (other session's result, entry supplied by that session; B9 continued from
3,000 to 5,000 with the base head's learning rate at 0.1 times): health recovers, the base keeps the variety,
but the decoded shapes are not more varied.** At 5,000, against B7 / J4 at 5,000: base share of field
variance 0.80 (B7 0.03); radius over 1,024 codes 0.10 to 0.31; clean 0.94 / 1.00 / 0.94, live 8.0 / 8.2 / 6.25,
anchors 0.79 / 0.83 / 0.80 (B9 at 3,000: 0.81 / 5.2 / 0.67). Pairwise (96 codes): 1 - overlap to the nearest
shape 0.445 / 0.471 / 0.475 (+/- 0.006), surface distance to the nearest 0.0690 / 0.0678 / 0.0688, volume
spread 0.25 / 0.16 / 0.19, half-thickness median 0.252 / 0.276 / 0.244. Picture
`figures/data_free_box_base_B10_5000.png`: thin plate and bar bases come out as chunky blobs; the correction
pads them out (outward bound 0.35), so the variety moved into the base without adding variety to the shapes.

**B8 and B9 finished (other session's result, entry supplied by that session; base-variation regularisers,
each B6 continued from 1,000 to 3,000; control B7 at 3,000).**

| At 3,000 | B7: none | B8: field gradients into the base x0.1 | B9: base spread push |
|---|---|---|---|
| Base share of field variance | 0.034 | 0.061 | 0.64 |
| Radius over 1,024 codes (5 to 95%) | about 0.40 | about 0.39 | 0.11 to 0.34 |
| Largest extent median / largest aspect | 0.16 / 1.4 | 0.17 / 1.5 | 0.28 / 10.4 |
| Clean / live / anchors | 0.94 / 6.5 / 0.74 | 0.94 / 7.5 / 0.77 | 0.81 / 5.2 / 0.67 |
| 1 - overlap, mean pair / to nearest (96 codes) | 0.671 / 0.479 | 0.627 / 0.433 | 0.709 / 0.465 |
| Surface distance to nearest | 0.0686 | 0.0655 | 0.0642 |
| Volume / area spread between shapes | 0.17 / 0.10 | 0.16 / 0.09 | 0.35 / 0.18 |
| Half-thickness median | 0.252 | 0.254 | 0.263 |

- B9's base keeps a job (rods, plates, boxes); shapes vary twice as much in size and area and the average
  pair differs more, but nearest neighbours are no further apart, shapes are not thinner, and validity, live
  directions and anchors are worse. B8 only slowed the fade. Picture `figures/data_free_box_base_B9_3000.png`.

**B7 finished (other session's result; B6 continued from 1,000 to 5,000, deviation budget widened to 0.5):
the base fades back to one shared rounded cube.** Base share of the field variance 0.16 / 0.07 / 0.03 / 0.03
at 1,000 / 2,000 / 3,000 / 5,000; over 1,024 codes at 5,000 the radius is 0.39 to 0.43 and the largest extent
0.08 to 0.13, with the per-code spread falling from 0.041 to 0.029. The budget never bound (deviation peaked
at 0.041 at 1,500, 0.029 at the end, tolerance 0.25).

| At 5,000 (96 codes, `variety_pairs.py`) | J4 | B7 |
|---|---|---|
| 1 - overlap to the nearest shape | 0.475 (+/- 0.005) | 0.471 |
| Surface distance to the nearest shape | 0.0688 | 0.0678 |
| Half-thickness, median | 0.244 | 0.276 |
| Run checks: clean / live / anchors (worst) | 0.94 / 6.25 / 0.80 (0.44) | 1.00 / 8.2 / 0.83 (0.51) |

- Same variety, slightly thicker, healthier. The base does not keep a job once the correction can make every
  shape. Picture `figures/data_free_box_base_B7_5000.png`. That session's next candidates (owner to decide):
  a weak cost on the correction's size near the surface, or anchor-box targets for the head.

**Base-head ablation finished 03:58 CST 15 Nov (other session's result; all from scratch, learned
rounded-box base, 1,000 updates; J4 at 1,000 for reference).**

| At 1,000 | B5: linear head on side features | B4: own network on the code | B6: side features to 64 tanh to 4 | J4 |
|---|---|---|---|---|
| Base share of field variance | 0.102 | 0.159 | 0.161 | - |
| Radius over 1,024 codes (5 to 95%) | 0.41 to 0.43 | 0.37 to 0.43 | 0.38 to 0.41 | - |
| Largest extent (median) | 0.10 | 0.125 | 0.128 | - |
| Clean / live / anchors | 0.94 / 3.75 / 0.53 | 0.94 / 4.0 / 0.56 | 1.00 / 5.25 / 0.58 | 0.94 / 5.0 / 0.54 |
| 1 - overlap to nearest shape (96 codes) | 0.282 | 0.289 | 0.349 | 0.310 (+/- 0.007) |
| Surface distance to nearest | 0.0437 | 0.0437 | 0.0492 | 0.0446 |
| Half-thickness median | 0.301 | 0.312 | 0.245 | 0.248 |

- B6 uses its base as much as B4, is the most varied by every pair measure (nearest-shape overlap distance
  12% above J4) and is the healthiest; B4 and B5 make shapes fatter. Pictures
  `figures/data_free_box_base_{B4,B5,B6}_1000.png`.

**Is the variety figure unfair to thin shapes? (owner's question, measured about 02:20 CST 15 Nov,
`analysis_scripts/variety_pairs.py`, 96 random codes, no code derivatives.)** Partly yes.

| At 5,000 updates | J4 (sphere base) | S6 (no base) |
|---|---|---|
| Total variance figure (surface motion per code step) | 1.09 | 0.77 |
| 1 - overlap, mean over pairs / to the nearest other shape | 0.66 / 0.47 | 0.73 / 0.47 |
| Surface distance, mean over pairs / to the nearest other shape | 0.116 / 0.069 | 0.109 / 0.058 |
| Field difference (the training's measure), mean / nearest | 0.123 / 0.072 | 0.112 / 0.055 |
| Half-thickness, median (thinnest tenth) | 0.244 (0.208) | 0.205 (0.160) |
| Spread between shapes: volume / area / compactness | 0.19 / 0.10 / 0.07 | 0.39 / 0.18 / 0.14 |

- The variance figure puts S6 30% below J4. By surface distance between pairs of shapes it is 6% below (15%
  to the nearest shape); by overlap it is level or ahead; and its shapes vary twice as much from code to code
  in volume, surface area and compactness (kind of shape).
- The variance figure counts surface motion in absolute distance, so a thin shape changing thickness counts
  for little. Scaled by thickness squared the two models come out equal (0.77 x (0.244 / 0.205)^2 = 1.09);
  that is an illustration from medians, not a measurement.
- The field difference, which the training's variety push uses, is the measure that rates thin shapes
  lowest (24% lower to the nearest shape).
- Same pattern at 2,500 (J4 against S5) and J7 at 11,000 is close to J4 at 5,000 on all of them.

**Unseen-object fit of the no-base model (S6 at 5,000; finished 03:47 CST 15 Nov): worse than J4 on every
measure and on every third of the objects, including the thinnest.** Same 225 Thingi objects, search and
scoring as J4 and J7.

| | J4 at 5,000 | J7 at 11,000 | S6 at 5,000 (no base) |
|---|---|---|---|
| Overlap, all / thinnest third | 0.809 / 0.722 | 0.815 / 0.728 | 0.768 / 0.676 |
| Surface distance, all / thinnest third | 0.0213 / 0.0258 | 0.0207 / 0.0251 | 0.0264 / 0.0310 |
| F-score at 0.02 / 0.04 | 0.62 / 0.87 | 0.64 / 0.88 | 0.51 / 0.80 |
| Anchors at their own codes | 0.81 | 0.88 | 0.64 |

- So the thinner shapes of the no-base model do not help it reach thin real objects, and the extra variety
  that pairwise measures credit it with does not cover real objects better.

**Code dimensions by layer (owner's question, measured 03:00 CST 15 Nov, `analysis_scripts/layer_dims.py`,
chart `figures/data_free_layer_dims_no_base_vs_J4.png`):** the code's 63 directions reach the main network
almost intact in both models (about 50 effective at the scale-and-shift output). In the no-base model the
scale and shift of main layer 1 carry only about 16 (J4 53), so layer 1 sees 13 (J4 49); J4 narrows gradually
through the main network instead. At the surface: effective 4.2 (no base) against 5.5 (J4), and directions
above 1% of the strongest 28 against 62. For the no-base model these fell from 33 to 15 between updates 500 and
1,000, while the anchor pull ramped in, then recovered steadily to 28 by 5,000; main layer 1 stayed at 12 to 13
from 2,000 on.

**B5 finished 02:05 CST 15 Nov (other session's result; from scratch, learned base head on the side network,
deviation budget): the base does get used from scratch.** Base share of the field variance 0.017 / 0.045 /
0.042 / 0.102 at 250 / 500 / 750 / 1,000 (B3, warm start: 0.0015 at 1,000), still rising. Over 1,024 codes
at 1,000: radius 0.41 to 0.43, largest extent 0.07 to 0.13 (median 0.10). Mostly a shared change (every base
boxier: radius 0.50 to 0.42, extents up), with the per-code spread growing (0.004 at 100 to 0.032). The
deviation budget is never binding after about 300. The rest is level with J4 at 1,000 (clean 0.94 / 0.94,
live 3.75 / 5.0, probe distance 0.097 / 0.097, anchors 0.53 / 0.54); rougher at 250. Picture
`figures/data_free_box_base_B5_1000.png`: the bases are fat rounded boxes whose stretch follows the decoded
shape's, but the decoded shapes are much smaller than their bases.

**S6 finished 00:57 CST 15 Nov (S5 continued to 5,000): the decoder without the fixed sphere trains, but at
5,000 updates it is behind J4 on every model measure, and the advantage it was meant to give has mostly gone.**

| At 5,000 updates | J4 (sphere as fixed base) | S6 (no base) |
|---|---|---|
| Fresh clean / 128 | 128 | 90 (119 at 4,000) |
| Total variance, 96 codes | 1.09 +/- 0.05 | 0.77 +/- 0.03 |
| Solid directions | 31 | 15 |
| Anchors (worst) | 0.81 (0.48) | 0.64 (0.08); 0.46 at 4,000 |
| Consistency miss, random codes | 0.035 | 0.041 |
| Slope error within 0.1 of the surface, random codes | 0.196 | 0.200 |
| Surface near side-on / past side-on | 24% / 1.2% | 24% / 2.9% |
| Slope error: outward / tilted / near side-on / past side-on | 0.12 / 0.17 / 0.35 / 0.55 | 0.17 / 0.18 / 0.29 / 0.48 |

- By 5,000 the fixed-base model has as much side-on surface as the no-base one; the no-base model keeps
  only a little more surface past side-on and a slightly milder rise of slope error with facing.
- Variety kept rising (0.57 at 2,500, 0.62 / 0.58 / 0.77 at 3,000 / 4,000 / 5,000) but stays about 30% under
  J4's, in half as many directions. Validity and anchor fit swing between checkpoints.
- Shapes (`figures/data_free_no_base_S6_vs_J4_5000.png`, same 8 codes; 64 codes in
  `figures/data_free_random_shapes_S6_5000.png`): thinner and more open than J4's, with fins, blades and thin
  webs between lobes; the defective ones are pinched apart or have a through-hole.
- Field chart `figures/data_free_field_sphere_start_S6_update5000.png`. Dead units in the last two hidden
  layers: 19% and 29%.

**B3 at 250 and 500 (other session's report, about 01:18 CST 15 Nov): the base does not move.** The head is
alive (no flat region) and the deviation budget is wide open (its weight decayed from 1,000 to the 0.01
floor), but over 1,024 fresh codes the radius stays 0.492 to 0.498 and every extent within +/-0.008; the
base's share of the field variance is 0.00. The rest of the model is healthy (at 250: clean 1.00, live 7.2,
anchors 0.77; J6 at 5,250: 1.00 / 6.25 / 0.74). That session's reading: from a warm start the correction
already makes every shape, so moving the base gives the loss nothing.

**B5's set-up (ran 01:27 to 02:05):** `B5_scratch_box_linear_head_budget02_1k` (other session; owner's rule).
B3's base design (linear head on the side network, deviation budget 0.01 to 0.2 over 500 updates) trained
from scratch on the J4 recipe, 1,000 updates; control J4 at 1,000. Question: does the base get used when
shapes first form? Decision script log `runs/data_free_latent/queue_logs/launch_after_B3_decide.out`.

**Superseded (never ran; replaced by the from-scratch B4 above):** `B4_J4_5000_box_linear_sepnet_budget02_1k` (owner's request; session
"decoder-shape-residual-parameterization", entry supplied by that session). B3 with the base coming from its
own small network on the code (code -> 64 tanh -> 4, last layer starting at zero), sharing nothing with the
side network, so the base cannot compete with the scales and shifts for the side network's features.
Everything else as B3 (linear output = deviation from the sphere, deviation budget 0.01 to 0.2 over 500
updates, J4 at 5,000 warm start, 1,000 updates). Branch `rounded-box-base` commit `05b384f`. Compare with B3
and J6 at 6,000.

**`B3_J4_5000_box_linear_head_budget02_1k`** (owner's design; session
"decoder-shape-residual-parameterization", entry supplied by that session; replaces the cancelled B2).
- The learned head again (no fixed map), redone: its output is the base's deviation from the sphere
  (extents b = u, radius r = 0.5 + u_r; no squash function, so it cannot go flat as B1's did).
- Hard limits only: radius between 0.05 and the reach, corner capped at 0.55; negative extents (a lens)
  cost weight 100.
- How far the base may deviate is set by a cost, not a schedule: an adaptive constraint `base_deviation`
  (mean over codes of |b|^2 + (r - 0.5)^2) whose tolerance, the budget squared, opens from 0.01^2 to 0.2^2
  over 500 updates; its weight starts at 1,000 and adapts like every constraint weight.
- J4 at 5,000 warm start (field identical at update 0), 1,000 updates, branch `rounded-box-base` commit
  `1cad02d`. Control: J6 at 6,000.
- Measured beforehand (one real update, head at zero): the shared push on the extents is downward, mostly
  from the anchor pull, then size and widen, against thicken. So the first thing to watch is whether the
  bases spread between codes or only shrink together.

**CANCELLED before it started (owner's decision, about 00:13 CST 15 Nov: no fixed map; no run folder exists): `B2_J4_5000_box_fixed_map_anchorfit_ramp500_2k`** (session
"decoder-shape-residual-parameterization", entry supplied by that session).
- The base is a rounded box that is a fixed function of the code: four fixed random code directions set the
  three extents (each axis on its own, so a rod can lie along any axis) and the rounding radius. Over random
  codes that gives 47% blob-like, 24% rods, 23% sheets, 6% boxes, median rounding radius 0.13.
- Every anchor gets a rounded box fitted to it, and the four base components of its code are set to match
  (limited to +/-2.5; mean code move 0.35 chord).
- J4 at 5,000 warm start, bases ramped in over 500 updates, 2,000 updates. Branch `rounded-box-base`, commit
  `73b3107`. Control: J6 at 7,000. Launcher log `runs/data_free_latent/queue_logs/launch_B2_box_fixed_map.out`.

**B1 finished about 23:40 CST 14 Nov (other session's result): the learned rounded-box base never left the
sphere.** `B1_J4_5000_box_base_ramp200_1k`: every code ended with the same base (radius 0.504, extents about
-0.02; the head drifted into the flat part of its soft ramp within about 100 updates), and the base's share
of the field variance was 0.00. So B1 is in effect J4 continued with four extra input features; at 1,000
updates it is level with J6 at 6,000 (clean 0.94 against 1.00, anchors 0.77 against 0.77, probe distance
0.117 against 0.123, live 4.5 against 7.25).

**S5 finished 23:00 CST 14 Nov: the first decoder without the fixed sphere that trains properly. Field as
accurate as J4's, variety and anchors behind but rising, validity clearly worse.**
`S5_J4_recipe_sphere_start_no_base_anchor_hold_surface_slope_2500`: fitted sphere start, small code spread
(`film_init_std=0.005`), outer limit by minimum, anchor pull off for 500 updates then ramped in over 500,
slope term counting the near-surface samples (`eikonal_all_points`), normal learning rate. Commit `082db17`.

| | J4 at 1,000 | S5 at 1,000 | J4 at 2,500 | S5 at 2,500 |
|---|---|---|---|---|
| Fresh clean / 128 | 126 | 46 | 128 | 76 |
| Total variance, 96 codes | 0.50 | 0.24 | 0.72 | 0.57 |
| Solid directions | 20 | 9 | 30 | 12 |
| Anchors | 0.54 | 0.24 | 0.71 | 0.52 |
| Consistency miss, random codes | 0.032 | 0.032 | 0.032 | 0.037 |
| Slope error within 0.1 of the surface, random codes | 0.161 | 0.166 | 0.165 | 0.185 |
| Surface facing past side-on | 1.0% | 26.5% | 0.4% | 6.7% |
| Slope error, outward-facing / near side-on surface | 0.10 / 0.29 | 0.13 / 0.24 | 0.09 / 0.33 | 0.16 / 0.23 |

- No collapse at the start. Trainer's field error 0.037 to 0.053 through update 1,250, then 0.038 to 0.039
  to the end (J4: 0.028); topology terms off 22% of updates after 1,250 (S1 to S4: 63 to 85%); slope weight
  off its ceiling (about 110 to 130) after 1,250.
- Still rising when it stopped: variance 0.026 / 0.24 / 0.46 / 0.57 at 500 / 1,000 / 2,000 / 2,500; anchor
  overlap in the training batches 0.26 / 0.37 / 0.44 / 0.47 (the pull reached full weight only at 1,000).
- Shapes (`figures/data_free_random_shapes_S5_2500.png`): limbs, necks and dumbbells, more of them than J4
  gives; some pinched in two. The 52 defective codes are several pieces or a through-hole.
- Field charts: `figures/data_free_field_sphere_start_S5_update{1000,2500}.png`.

**Why B1's learned base stayed a sphere (checked about 00:10 CST 15 Nov from its checkpoint,
`analysis_scripts/base_head_check.py`):** all three extent outputs of the head sit 6 to 7 temperatures into
the flat part of its ramp (temperature 0.01) for every one of 2,000 random codes, where under 0.5% of any
gradient gets through; it was already there at update 250. The outputs are nearly the same for all codes
(the head's inputs are non-negative with a shared mean, and it starts at zero, so the first steps move every
code's base together). One optimiser step can move an output by about as much as the ramp's temperature.
The owner said (about 00:05) that a base that is a fixed function of the code, as in B2, defeats the point; the
other session has been told and asked to check with the owner before B2 starts.

**B1's set-up (other session's entry):** `B1_J4_5000_box_base_ramp200_1k` (owner's
request, session "decoder-shape-residual-parameterization"; entry supplied by that session).
- The decoder's fixed sphere is replaced by a rounded box set by the code: half-extents b1 >= b2 >= b3 and
  a rounding radius r from a new 4-output head on the side network, zero at the start, so every code starts
  as the radius-0.5 sphere. It is an exact signed distance; b = (L, 0, 0) is a rod, (a, a, 0) a sheet, small
  r a box. The corner reach |b| + r is capped at 0.55 (bound 0.9 minus the correction bound 0.35).
- The main network also gets the base field and the nearest point of the core box as inputs
  (zero-initialised columns).
- Warm-started from J4 at 5,000 (same field, anchor codes and multipliers), 1,000 updates, with
  `base_shape=rounded_box` and `base_ramp_steps=200`.
- Branch `rounded-box-base` (worktree `.claude/worktrees/rounded-box-base`, commits `a0e1f09`, `9474382`);
  not on the data-free branch.
- Control: J6 at 6,000 (J4 continued with nothing changed: clean 1.00, anchors 0.77, live 7.25, probe
  distance 0.123).
- New diagnostics: base/r, b1 to b3, elongation, rod and sheet fractions, and base/field_variance_share
  (variance of the base field alone over that of the whole field; near 1 means the variety lives in the base).
- Launcher log: `runs/data_free_latent/queue_logs/launch_B1_box_base.out`.

**S4 finished 21:22 CST 14 Nov (S1 with the outer limit by minimum, commit `2a36fe0`): the far
field is repaired, but the run is not better.** `S4_J4_recipe_sphere_start_no_base_outer_min_1k`.

| At 1,000 updates | J4 (sphere base) | S1 (no base, blend) | S4 (no base, minimum) |
|---|---|---|---|
| Fresh clean / 128 | 126 | 128 | 31 (97 with a handle) |
| Total variance, 96 codes | 0.50 | 0.14 | 0.05 |
| Solid directions | 20 | 10 | 7 |
| Anchors | 0.54 | 0.37 | 0.29 |
| Consistency miss, random codes | 0.032 | 0.063 | 0.040 |
| Slope error, random codes (all / near the surface) | 0.094 / 0.161 | 0.151 / 0.211 | 0.111 / 0.223 |
| Slope weight (ceiling 200) | 25 to 40 | 200 | 200 |

- Charts `figures/data_free_field_sphere_start_S4_update{25,1000}.png`: away from the surface the field now
  follows the shape (no pinned ring). At the surface itself the slope is far too steep (a thin rim near 2).
- Every shape became a ring: two lobes joined by a pinch in the slice, a through-hole in 3-D. Nothing in the
  loss removes a through-hole.
- The early collapse is still there and is caused by the anchor pull. 25 updates from a fresh start
  (`early_collapse.py`): the share of the ball inside the shapes goes 0.173 to 0.005 as run, to 0.013
  without the variety push, to 0.243 without the anchor pull, and stays 0.174 with neither. The fixed-base
  J4 recipe shrinks too, but only to 0.075.
- 100-update scan of starts for the no-base decoder (share of the ball inside the shapes, start 0.17; field
  error): as S4 0.095 / 0.070; usual code spread 0.083 / 0.061; usual spread with anchors ramped from 0 over
  300 updates 0.031 / 0.188 (centre of the shapes outside); usual spread, no anchors 0.021 / 0.095 (centre
  outside); small spread, no anchors 0.163 / 0.039. So only the small spread without the anchor pull keeps a
  healthy start; with the usual spread the variety push alone hollows the shapes out.
- Likely reason: at the start all codes give the same field, and the anchors are 96 very different objects,
  so the pull moves every code toward the average anchor field, which is inside only in a small core.

**Field contour charts of the sphere-start runs (owner's request, 20:50 CST 14 Nov) found a flaw in the
no-base decoder as first built: its outer blend pins the far field.** To keep shapes inside radius 0.9 the
network's output was blended into the sphere's field R - rho between radius 0.7 and 0.9. That forces the
field there to describe a sphere of radius 0.5 whatever the shape is.

- S1 after 25 updates: every shape has shrunk to a rounded square about 0.17 in half-width; the far field
  still holds the radius-0.5 sphere's values, so between them the slope is about 0.5 (mean error 0.16 over
  the slice, 0.02 near the surface). J4 after 25 updates also shrinks (radius about 0.35) but its whole
  field follows (mean error 0.01).
- S1 at 1,000: the mismatch remains (slope too flat in a ring outside the surface, too steep inside).
  S3 at 1,000: shapes collapsed into small pieces in an axis-aligned pattern, same pinned ring.
- So the slope term could never be satisfied, which is why its weight sat at the ceiling and it took over
  the update. The conclusion drawn from S1 to S3 (the fixed sphere does more good than harm) is not
  established.
- Charts: `figures/data_free_field_sphere_start_S1_update{0,25,500,1000}.png`,
  `..._S2_update1000.png`, `..._S3_update{300,1000}.png`, `figures/data_free_field_fixed_base_J4_update{25,1000}.png`
  (script `analysis_scripts/field_contours3.py`; three rows: contours against true distance, error, slope).
- Fix (commit after `394cccb`): `outer_limit="min"`, the smooth minimum of the network's output and the
  bounding ball's field. S4 is S1 with only that changed. First 100 updates: field error 0.047 / 0.059 /
  0.042 at updates 25 / 50 / 100, against S1's 0.157 / 0.20 / 0.074.

**S3 finished 20:40 CST 14 Nov (second sphere-start try, owner's request): clearly worse than the first.**
`S3_J4_recipe_sphere_start_usual_spread_anchor_ramp_1k` is S1 with the code's usual starting spread
(`film_init_std` 0.1) and the anchor pull ramped in from 1% to full weight over 300 updates
(`anchor_weight_ramp_steps=300`, commit `394cccb`).

| At 1,000 updates | J4 (sphere base) | S1 (no base) | S3 (no base, usual spread, anchor ramp) |
|---|---|---|---|
| Fresh clean / 128 | 126 | 128 | 5 |
| Total variance, 96 codes | 0.50 | 0.14 | 0.08 |
| Solid directions | 20 | 10 | 11 |
| Anchors | 0.54 | 0.37 | 0.10 |
| Trainer's field error, updates 250 to 1,000 (median) | 0.028 | 0.054 | 0.07 |
| Updates with the topology terms switched off | 46 | 633 | 849 |
| Consistency miss, random codes | 0.032 | 0.063 | 0.181 |
| Size (target 0.156) | 0.141 | 0.136 | 0.096 |

- Shapes broke up (handles and several pieces in nearly every code from update 250 on), shrank and drifted
  off-centre (mean offset 0.054 against 0.017).
- The field broke by update 25 although the anchor weight was under 10 there, so the anchor pull was not
  what knocks the field off first; with the wider starting spread the variety push does it. With the field
  that inaccurate the topology terms stayed off for 85% of the run, which is why broken shapes were never
  repaired.
- At update 500 the largest part of the update was the barrier term at its weight ceiling (the slope near the
  surface collapsing below one half), working against the slope term.
- Pictures: `figures/data_free_random_shapes_S3_1000.png`, `figures/data_free_anchor_fit_S3_1000.png`.

**S2 finished 20:04 CST 14 Nov (S1 at one third of the learning rate, owner's request): the lower rate
changed almost nothing.** `S2_J4_recipe_sphere_start_no_base_lr_third_1k`, lr 2.121e-4.

| At 1,000 updates | J4 (sphere base) | S1 (no base) | S2 (no base, lr / 3) |
|---|---|---|---|
| Fresh clean / 128 | 126 | 128 | 128 |
| Total variance, 96 codes | 0.50 | 0.14 | 0.11 |
| Solid directions | 20 | 10 | 9 |
| Anchors | 0.54 | 0.37 | 0.34 |
| Trainer's field error, updates 250 to 1,000 (median) | 0.028 | 0.054 | 0.053 |
| Updates with the topology terms switched off | 46 | 633 | 495 |
| Slope weight (ceiling 200) | 25 to 40 | 200 | 200 |
| Consistency miss, random codes | 0.032 | 0.063 | 0.070 |
| Slope error within 0.03 of the surface: outward / tilted / side-on | 0.10 / 0.15 / 0.29 | 0.39 / 0.32 / 0.17 | 0.19 / 0.19 / 0.21 |

- The field error sits at about twice J4's at either learning rate with the slope weight pinned at its
  ceiling, so the rate is not the cause.
- What the change was meant to fix did go away: without the base, slope error no longer rises as the
  surface turns away from facing outward. But the error is higher overall, variety is a quarter of J4's and
  anchors fit worse, so as tested it is a net loss.
- Pictures: `figures/data_free_random_shapes_S2_1000.png`, `figures/data_free_anchor_fit_S2_1000.png`.

**Why the sphere-start runs lag (measured 20:25 CST 14 Nov with `analysis_scripts/grad_by_term.py`, which
takes the gradient of every weighted loss term inside one real update):**

| Share of the update's direction | J4 at 1,000 | S1 at 500 | S1 at 1,000 | S2 at 1,000 |
|---|---|---|---|---|
| Slope (eikonal) term | -6% (weight 29) | +110% (weight 200) | +80% (200) | +149% (200) |
| Anchor pull | +87% | -13% | -14% | -90% |
| Variety push | +21% | -7% | -2% | -18% |
| Size and position | -3% | +6% | +34% | +36% |

- With the sphere as a fixed base the update follows the anchor pull and the variety push. Without it the
  slope term, at its weight ceiling, sets the direction and the anchor pull and variety push are overruled
  (negative: the update moves against them on balance).
- At update 0 the anchor pull is 89% of the gradient in S1 (70% in J4), so it is the first thing to push the
  field off; the variety gradient starts 20 times smaller than J4's (21 against 412) because the code's
  starting spread was cut 20-fold.
- Both runs were still gaining variety when they stopped (S1 0.067 at 500, 0.137 at 1,000; S2 0.028, 0.110),
  but the anchor fit and the field error were flat.

**S1 finished 19:45 CST 14 Nov: training without the fixed sphere, from the fitted sphere, stays valid but
the shapes barely leave the sphere and the field is worse.** `S1_J4_recipe_sphere_start_no_base_1k` is the J4
recipe for 1,000 updates with `sphere_base=False` (commit `5efc02e`): the main network's own output is the
field, blended into the sphere's field between radius 0.7 and 0.9; main network started from the fitted
sphere; code's starting spread 0.05 times the usual (`film_init_std=0.005`); same learning rate.

| At 1,000 updates | J4 (sphere as fixed base) | S1 (no base, fitted start) |
|---|---|---|
| Fresh clean / 128 | 126 | 128 |
| Total variance, 96 codes | 0.50 +/- 0.02 | 0.14 +/- 0.004 |
| Solid directions | 20 | 10 |
| Anchors | 0.54 | 0.37 |
| Slope error near the surface, random codes | 0.16 | 0.21 (0.36 within 0.03 of it) |
| Consistency miss, random codes | 0.032 | 0.063 |
| Consistency miss, anchor codes | 0.041 | 0.183 |
| Surface facing outward / tilted / near side-on | 35% / 52% / 13% | 67% / 32% / 1% |

- Shapes are lumpy near-spheres (picture `figures/data_free_sphere_start_S1_vs_J4_1000.png`, 64 codes in
  `figures/data_free_random_shapes_S1_1000.png`). The aim was more surface that does not face outward; at
  1,000 updates there is less of it, because the shapes have moved less.
- The field broke within the first 25 updates (slope term 0.0007 to 0.09, consistency 0.003 to 0.17) and never
  settled: the trainer's field error swung between 0.04 and 0.1, so the topology terms were off most of the
  run. S2 shows the same jump by update 50 at a learning rate of under 1e-4, so the learning rate is not
  what causes it.
- Validity held throughout (all 16 check codes clean at every check, 128 of 128 at the end).

**Sphere fit finished 19:23 CST 14 Nov (owner's request; `runs/data_free_latent/S0_sphere_fit_20261114/`):
the plain main network reproduces a sphere closely in under three minutes.** Why: with the sphere as a fixed
base, wherever a surface does not face away from the centre the network has to cancel the sphere's slope
first, and measured slope errors follow that order (side-on surfaces about twice the error of outward-facing
ones; no surface at random codes faces the centre).

| 4,000 updates, 160 s | 3 Fourier bands | No Fourier bands |
|---|---|---|
| Surface radius (target 0.5) | 0.4998 +/- 0.0003 | 0.4997 +/- 0.0003 |
| Value error near the surface / rest of the ball | 0.0003 / 0.0007 | 0.0003 / 0.0007 |
| Slope error near the surface / rest of the ball | 0.027 / 0.042 | 0.013 / 0.012 |
| Slope error at the centre (cone tip) | 0.12 | 0.16 |
| Stray pieces | none | none |

- With the code switched on, all 32 random codes are clean at every starting spread tried. At the usual
  spread codes start 19 times further apart than today (0.021 against 0.0011); 0.05 times the usual matches.
- The fitted final layer is small (length 0.044; J4 at 1,000: 0.09), and a learning-rate-sized nudge of all
  weights moves the field 0.7 times as much as at today's start (hidden layers alone: 10 times as much).
- Pictures: `figures/data_free_sphere_fit_fourier3.png`, `figures/data_free_sphere_fit_fourier0.png`.
- The first launch of the fits died with the session restart at 19:14 (empty logs); a memory scan then found
  56 damaged cached files (dropped; disk copies clean). The fault is active again.

- Background: the actual Adam step (`analysis_scripts/step_by_layer.py`) gets only 8 to 20% of its shape
  change from the final layer, though the final layer has 97% of the raw gradient; the side network gives
  about half. But the final layer's weights are tiny, so each step changes them by 10 to 30 times more in
  relative terms than any other layer, and that change is mostly the same for every code (shifts all
  shapes at once). Suspected source of the late swings (J6's final checkpoint: 17 defective codes of 128).
- New switches (commit `eec4a4b`, all off by default): `out_lr_scale` (final layer's learning rate
  as a fraction of the rest), `lr_decay_start` (hold the rate until then, then cosine to the floor),
  `weight_average_decay` (running average of the weights; does not affect training; every save also
  writes `checkpoint_average_last.pt`).
- L1: J4 from 5,000 to 10,000 with `out_lr_scale=0.1`, nothing else changed. L2: the same plus decay from
  5,000 to a tenth of the rate at 10,000 (`lr_decay_start=5000`, `lr_floor_fraction=0.1`).
- Both keep a weight average (`weight_average_decay=0.998`, about 500 updates). Training is identical with
  or without it, so each run gives the live model and the averaged model.
- Control for both: J6 (J4 continued from 5,000 to 10,000 with nothing changed; measured at 6,000, 7,500,
  9,000 and 10,000). Checkpoints and averaged checkpoints are kept at the same updates.
- Comparisons: final-layer rate = L1 against J6; decay = L2 against L1; averaging = each run's averaged
  model against its live model.

**W1 finished 16:47 CST 14 Nov (weight-normalisation ablation, owner's request): worse than the control on
every measure.** `W1_J4_recipe_weight_norm_1k` is the J4 recipe from scratch for 1,000 updates with
`weight_norm=True` (commit `a9fa1ce`, off by default): each unit of the main network stores its weights as
a direction and a size. No crease term.

| At 1,000 updates | J4 (control) | W1 (weight normalisation) |
|---|---|---|
| Fresh clean / 128 | 126 | 57 (most in pieces) |
| Anchors | 0.54 | 0.25 |
| Solid directions | 20 | 12 |
| Total variance, 96 codes | 0.50 +/- 0.02 | 0.53 +/- 0.04 (codes range 0.05 to 1.97) |
| Slope error near the surface, random codes | 0.161 | 0.229 |
| Consistency miss, random codes | 0.032 | 0.056 |
| Dead units, last hidden layer | 10% | 17% |
| Output layer's weight length | 0.090 | 0.0015 (0.0006 at update 500) |
| Hidden layers' weight-row length | 1.43 to 1.47 | 1.40 to 1.41 (did not move) |

- The run was unstable: 9 of 16 check codes clean at update 250, all 16 at 500 and 750, 5 at 1,000.
- Why (from the saved weights): the output layer's size became one number for the whole network. It fell
  about 60-fold, to where a single optimiser step is as large as the number itself, so the size of every
  shape's departure from the sphere swung from update to update. The hidden units' sizes, at 1.41, moved by
  under 1% in 1,000 updates, so they were in effect frozen.
- The imbalance it was meant to remove got more extreme: the output layer's weights are now tiny (typical
  0.0001) and the last hidden layer's values large (3.8 against 0.09).
- Pictures: `figures/data_free_random_shapes_W1_1000.png`, `figures/data_free_anchor_fit_W1_1000.png`.

**C3 finished 16:14 CST 14 Nov: at weight 2 the crease term does not sharpen anything, and 44 of 128 fresh
codes have a through-hole.** `C3_J4_recipe_crease2_1k` is C1 with weight 2 in place of 10.

| At 1,000 updates | J4 (no term) | C3 (weight 2) | C1 (weight 10) |
|---|---|---|---|
| Flat share of the surface | 1.8% | 2.3% | 38% |
| Median bend radius | 0.14 | 0.17 | 0.76 |
| Total variance, 96 codes | 0.50 +/- 0.02 | 0.51 +/- 0.02 | 0.32 +/- 0.01 |
| Solid directions | 20 | 20 | 9 |
| Fresh clean / 128 | 126 | 84 (44 with a handle) | 127 |
| Anchors | 0.54 | 0.54 | 0.40 |
| Slope error near the surface, random codes | 0.161 | 0.169 | 0.143 |

- Picture: `figures/data_free_crease_weights_0_2_10_at_1000.png` (same 8 codes at the three weights).
- No middle ground between 2 and 10 in these runs: at 2 the shapes are the same blobs as without the term
  (its own value rose during training, 0.58 to 0.70, so it was losing to the other terms); at 10 they are
  boxes. In C1 the switch happened abruptly between updates 200 and 300.
- The handles in C3 are well outside what the same recipe has given at 1,000 updates in other runs (126 to
  128 clean), so the term probably caused them, but that is one run.


**C2 finished 15:37 CST 14 Nov: on a mature model the term at weight 10 erased the limbs and detail. Shapes
became smooth lumps with a few flattened faces, and variety fell about nine-fold.**
`C2_J7_11000_crease10_to_12k` continued J7 from 11,000 to 12,000 at J7's learning rate plus the term at
weight 10. J7's own 12,000 is the same continuation without the term.

| | J7 at 12,000 (no term) | C2 at 12,000 (weight 10) |
|---|---|---|
| Flat share of the surface (bend radius over 1) | 1.2% | 11.5% |
| Median bend radius | 0.11 | 0.44 |
| Tight share (radius under 0.11) | 51% | 5% |
| Total variance, 96 codes | 1.07 +/- 0.05 | 0.12 +/- 0.01 |
| Solid directions | 26 | 13 |
| Fresh clean / 128 | (J7 at 11,000: 126) | 127 |
| Anchors (worst) | 0.84 | 0.85 (0.48) |
| Slope error near the surface, random codes | 0.36 | 0.16 |
| Consistency miss, random codes | 0.051 | 0.034 |

- Picture: `figures/data_free_crease_C2_vs_J7_12000.png` (same 8 codes: J7 at 11,000, J7 at 12,000, C2 at
  12,000); 64 random codes in `figures/data_free_random_shapes_C2_12000.png`.
- The anchors were not hurt (0.85) and the field at random codes is much more accurate, because the
  surfaces are now simple.
- Reading of C1 and C2 together: the term charges for every bit of curved surface, and the cheapest way to
  have less of it is a simpler shape. From scratch that gave boxes; on a mature model it removed limbs
  before it sharpened anything. At weight 10 it overpowers the variety push.

**C1 finished 15:00 CST 14 Nov: at weight 10 the term works, and too strongly. Every random shape becomes a
box with flat faces and sharp edges, and variety drops by a third.**
`C1_J4_recipe_crease10_1k` is the J4 recipe from scratch for 1,000 updates plus the term at weight 10.

| | J4 at 1,000 (no term) | C1 at 1,000 (weight 10) |
|---|---|---|
| Flat share of the surface (bend radius over 1) | 1.8% | 38% |
| Median bend radius | 0.14 | 0.76 |
| Tight share (radius under 0.11) | 35% | 20% |
| Total variance, 96 codes | 0.50 +/- 0.02 | 0.32 +/- 0.01 |
| Solid directions | 20 | 9 |
| Fresh clean / 128 | 126 | 127 |
| Anchors | 0.54 | 0.40 |
| Slope error near the surface, random codes | 0.161 | 0.143 |
| Consistency miss, random codes | 0.032 | 0.031 |

- Pictures: `figures/data_free_crease_C1_vs_J4_1000.png` (same 8 codes, both models, shaded) and
  `figures/data_free_random_shapes_C1_1000.png` (64 random codes).
- The boxes are lined up with the coordinate axes, and some have faces that bow inward to thin fins at the
  edges.
- The field at random codes is no worse (slightly better slope). The anchors fit worse (0.40 against 0.54),
  although the term is not applied to them.
- C3 repeats C1 at weight 2 to see whether a middle setting keeps the variety.

- The term (`crease_weight`, commit `371fb21`, off by default): at surface points it measures how fast the
  surface normal turns over a step of 0.02 along the surface, and costs it turn^2 / (turn^2 + 3^2). Flat
  surface is free and the cost levels off at 1, so one sharp edge is cheaper than a long gentle bend. It is
  0 on a plane, 0.41 on a ball of radius 0.4 and 0.04 on a cube of the same size. Anchor codes are left out.
- Before the test: on J4 and J7 the term reads 0.73 to 0.75, only 2% of surface pairs are flat, and the
  median bend is 6.2 to 6.7 (radius about 0.15). So current shapes are curved nearly everywhere.
- Weight 10 was chosen from gradient sizes: the term's gradient is about 5 per unit weight against about 100
  for the rest of the loss. The old smoothness term (`grad_change_weight` 0.001) has a gradient about a
  millionth of the whole, so it is not what rounds the shapes.
- C1 is the J4 recipe from scratch for 1,000 updates plus the term; compare with J4 at 1,000.
- C2 continues J7 from 11,000 to 12,000 at J7's learning rate plus the term; J7's own 12,000 is the exact
  control.
- The memory fault is still present (one damaged cached library file found and dropped at 14:20 before
  launch); both runs are short for that reason.

**Second trainable-code experiment finished 08:55 CST 14 Nov (forked session): the surface-band variety
measure cuts thin walls and gaps by a third to a half but costs about a quarter of the variety.**
`TA2_J4_trainable_codes_surface_band_5k` is TA1 (next entry) with `variety_surface_band=0.1` as the only
difference; `TA2s_settle_6k` is its settle stage.

| | TA1 at 5,000 | TA2 at 5,000 | TA1s at 6,000 | TA2s at 6,000 |
|---|---|---|---|---|
| Total variance, 96 codes | 1.06 | 0.78 | 1.02 | 0.76 |
| Solid directions (effective) | 29 (8.5) | 31 (9.1) | 31 (8.8) | 31 (9.7) |
| Fresh clean / 128 | 128 | 124 | 127 | 126 |
| Anchors (worst) | 0.79 (0.39) | 0.80 (0.42) | 0.88 (0.50) | 0.86 (0.53) |
| Thin-wall sites per pass | 252 | 163 | 323 | 176 |
| Gap sites per pass | 114 | 68 | 131 | 76 |
| Thicken multiplier | 1,000 (ceiling) | 585 | 1,000 | 687 |
| Consistency error | 0.038 | 0.036 | 0.041 | 0.037 |
| Unseen-object fit, 225 objects | | | 0.809 | 0.806 |

- The variance difference is several standard errors (about 0.03 to 0.045 each), so it is real. At update
  2,500 the band run was well behind (22 solid directions, anchors 0.60, against 30 and 0.69) and caught up
  on directions and anchors by 5,000.
- Random shapes with the band are chunkier and rounder; without it they have more thin limbs and flat parts.
  The fewer thin walls come largely from fatter shapes.
- Validity is no better with the band (4 defective codes of 128 at 5,000 against none; 2 against 1 after
  settling), and the unseen-object fit is the same (thin or hollow-centred objects 0.714 against 0.721).
- The earlier 1,000-update test (V1) had shown the same variety with half the thin walls; over a full run it
  is a trade.
- Pictures: `figures/data_free_random_shapes_TA1s_6000.png`, `figures/data_free_random_shapes_TA2s_6000.png`,
  and `data_free_anchor_fit_` for each.

**First trainable-code experiment finished 05:08 CST 14 Nov (forked session): trainable anchor codes make no
measurable difference; the settle stage improves the anchor fit.**

| Checkpoint | Total variance, 96 codes | Solid directions | Fresh clean / 128 | Anchors (worst) | Closest two anchor codes (cosine) |
|---|---|---|---|---|---|
| J4 at 5,000 (frozen codes) | 1.09 | 31 | 128 | 0.81 (0.48) | 0.00 (perpendicular) |
| TA1 at 2,500 | 0.97 on 24 codes (J4: 0.80) | 30 | 123 | 0.69 (0.07) | 0.19 |
| TA1 at 5,000 | 1.06 | 29 | 128 | 0.79 (0.39) | 0.48 |
| TA1s at 5,500 (settle) | 0.89 on 24 codes | 31 | 127 | 0.88 (0.53) | 0.49 |
| TA1s at 6,000 (settle) | 1.02 | 31 | 127 | 0.88 (0.50) | 0.49 |

- `TA1_J4_trainable_codes_5k` is exactly the J4 recipe with trainable anchor codes (fixed for 500 updates,
  then their learning rate ramps to 3e-4 by 1,500). `TA1s_settle_6k` continues it to 6,000 at one-tenth of
  both learning rates.
- Variety, directions and validity are the same as J4 within the noise (standard error of the 96-code
  variance about 0.045).
- The codes did move (one pair of anchors ended at cosine 0.48 from perpendicular) but nothing collapsed.
- The settle stage raised the anchor fit from 0.79 to 0.88 and brought the size back to target (0.152 against
  0.156), with no loss of variety; 500 updates of it were enough. The same happened in J7 with frozen codes
  (0.82 to 0.88), so the gain comes from the settle stage, not from the trainable codes.
- Unseen-object fit on TA1s at 6,000 (the same 225 Thingi objects): mean overlap 0.809, identical to J4 at
  5,000 (0.809; J7 at 11,000: 0.815). With a filled centre 0.879 (J4 0.882); thin or hollow-centred 0.721
  (J4 0.717), 70% of them at or above 0.70 (J4 64%); worst object 0.32 (J4 0.39). So neither the trainable
  codes nor the better anchor fit widened what the model can reach. Results: `followup/fit_TA1s_6000.json`.

**J7 finished about 01:12 CST 14 Nov (J6 continued from 10,000 to 12,000 at one-tenth learning rate): the
anchor fit improved a lot, validity recovered most of the way, variety held.**

| Update | Fresh clean / 128 | Anchors, all 96 (worst) | Total variance, 96 codes | Solid directions |
|---|---|---|---|---|
| 10,000 (start) | 111 | 0.82 (0.28) | 1.42 +/- 0.09 | 31 |
| 10,300 | 125 | 0.83 (0.31) | 0.91 on 24 codes | 26 |
| 10,500 | 121 | 0.87 (0.50) | 1.14 on 24 codes | 28 |
| 11,000 | 126 | 0.88 (0.52) | 1.27 +/- 0.06 | 31 |
| 12,000 | 122 | 0.84 (0.44) | 1.07 +/- 0.05 | 26 |

- Anchors at 11,000 by group: 64 thick 0.93 (J4 at 5,000: 0.87), all at or above 0.80; 16 thin rods 0.76
  (0.63), one in pieces (three); 16 thin sheets 0.82 (0.72). The field at anchor codes is 0.006 from the
  objects' true distance (J4: 0.013).
- Validity: from 17 defective codes of 128 to between 2 and 7. Better, not fully steady; the trainer's own
  probes still ranged from 0.75 to 0.94 clean.
- Field at random codes unchanged (slope error 0.26 near the surface, consistency miss 0.037). Dead units 89.
- The 11,000 checkpoint (`checkpoint_11000.pt`) is the best on anchors. Its 225-object unseen fit is only
  slightly ahead of J4's: overlap 0.815 (J4 0.809), surface distance 0.0207 (0.0213), share of surface within
  0.02: 0.64 (0.62), within 0.04: 0.88 (0.87); thin or hollow-centred objects 0.725 (0.717). So the better
  anchor fit did not widen what the model can reach beyond its anchors by much.

**Size of the anchors' neighbourhood (J7 at 11,000; picture `figures/data_free_anchor_neighbourhood_J7_11000.png`):**
moving the code away from an anchor's code, the decoded shape stays essentially the anchor out to about 25 to
30 degrees (overlap with the anchor object 0.88 at the code, 0.84 at 25 degrees, 0.81 at 30), is half-way to
an unrelated shape at about 50 degrees (cosine 0.63; thick anchors 53, thin rods 48, thin sheets 47), and is
indistinguishable from an unrelated random code by 75 to 90 degrees. That matches where training samples
around anchors (out to 60 degrees, pull down to 37% at 29). Random codes sit far outside: median 72 degrees
from the nearest anchor code, none of 200,000 closer than 52 degrees, 0.1% within 60.

**What J7 was:** `J7_J6_lower_lr_to_12k` continued J6 from its end (update 10,000) to 12,000 at one-tenth of the learning rate
(6.364e-5 in place of 6.364e-4), nothing else changed. J6's final checkpoint was unhealthy (17 of 128 fresh
codes defective, trainer's probe 0.62 clean), and an earlier one-tenth-rate stage repaired validity in another
run. Questions: does validity come back and stay steady, does the anchor fit stop swinging (0.73 to 0.82 in
J6), and does variety hold (J6: about 1.07 on 96 codes up to 7,500, 1.42 with broken shapes at 10,000)?
Checkpoints kept at 10,300, 10,500 and 11,000. The memory fault is still unfixed; the Python environment was
re-scanned clean just before launch.

**Queued behind J7 by the forked session (owner's request, 00:55 CST 14 Nov; about 6 hours in all; each
stage starts by itself when the one before ends, and all are detached from any session):**
1. `TA1_J4_trainable_codes_5k` (5,000 updates from scratch, about 2 h 20 min): exactly the J4 recipe (its
   saved settings are reused: 96 anchors, even-code penalty, no filled centre, leak 0.05, sharpness 100, three
   Fourier bands) with one change, **trainable anchor codes**: the codes are fixed for the first 500 updates,
   then their learning rate ramps to 3e-4 by update 1,500 (`anchor_embed_lr=3e-4`). In every run since G2 the
   anchor codes were frozen; the one earlier comparison (G1 against G2, old recipe) favoured frozen codes.
   Compare with J4 at the same updates (1,000: 0.585 / 21 solid / 126 clean / anchors 0.54; 2,500: 0.804 / 30 /
   128 / 0.71; 4,000: 0.985 / 32 / 120 / 0.76; 5,000: 1.118 / 31 / 128 / 0.81). Watch `anchor/lr` in the log
   (0 until update 500), `code cos max` in the probe lines (how close the two nearest anchor codes have moved;
   0.00 while they are perpendicular) and anchors reproduced. Checkpoints kept at 1,000, 2,500 and 4,000.
2. `TA1s_settle_6k` (about 30 min): TA1 continued from 5,000 to 6,000 at one-tenth of both learning rates
   (decoder 6.364e-5, anchor codes 3e-5), nothing else changed: the settle stage. Checkpoint kept at 5,500.
3. `TA2_J4_trainable_codes_surface_band_5k` (5,000 updates from scratch): TA1 again with the surface-band
   variety measure (`variety_surface_band=0.1`), the only difference from TA1. Its logged nearest-neighbour
   distance is on a different scale, so do not compare that number with TA1's. Compare everything else with
   TA1 at the same updates; the earlier 1,000-update test (V1) gave the same variety with about half the
   thin-wall and gap sites.
4. `TA2s_settle_6k`: TA2 continued to 6,000 the same way as TA1s.
These run on the fork's worktree (`.claude/worktrees/procedural-anchors`, which now holds the same code as the
main data-free branch plus launchers in `scripts/data_free_queue/`). Before each stage starts, the Python
environment, the Python installation, the code, the anchor sets and the run folder are compared with the
disk and damaged in-memory copies are dropped (result in the run folder's
`memory_scan_before_attempt_1.txt`). If the trainer dies before logging a single update (the memory fault's
usual symptom), the log is kept as `train_failed_1.log` and the stage is tried again, up to three times; a
stage being scanned or retried still shows as training. If a stage fails all three attempts the stages
behind it do not start. Queue logs: `runs/data_free_latent/queue_logs/`.

**Status at 00:05 CST 14 Nov: evaluations work again, without a restart.** A
scan found 24 files with damaged in-memory copies (21 in the Python environment, including torch's and
CUDA's main libraries, plus one checkpoint and two files of the third Thingi set); all were clean on disk and
were dropped from memory (`analysis_scripts/scan_cache.py --evict <folder>`). The machine has NOT been
restarted and the memory has NOT been fixed, so this will recur; every damaged word sits at the same position
inside a 64-byte block, which points at the memory hardware or its settings.

**J6 finished at 23:03 CST 13 Nov (J4 continued from 5,000 to 10,000): no gain from the extra 5,000 updates.**

| Update | Total variance | Solid directions | Fresh clean / 128 | Anchors (worst) | Dead units |
|---|---|---|---|---|---|
| 5,000 (J4) | 1.118 | 31 | 128 | 0.81 (0.48) | 88 |
| 6,000 | 1.138 | 30 | 126 | 0.76 (0.29) | not measured |
| 7,500 | 1.031 | 29 | 128 | 0.80 (0.37) | 63 |
| 9,000 | 0.799 | 25 | 127 | 0.73 (0.18) | 72 |
| 10,000 | 1.336 | 31 | 111 | 0.82 (0.28) | 79 |

**How noisy the variance figure is (measured 00:30 CST 14 Nov):** the table's total variance uses 24 random
codes; shapes differ a lot (per-code values 0.4 to 2.2), so its standard error is about 0.09, and four
separate groups of 24 codes report values up to 0.11 apart for the same model. Re-measured with 96 codes
(standard error about 0.045): 5,000: 1.09; 6,000: 1.09; 7,500: 1.06; 9,000: 0.84; 10,000: 1.42 (standard
error 0.09 there, with a few extreme codes). So 5,000 to 7,500 is flat, the dip at 9,000 is real, and the
rise at 10,000 comes with broken shapes. Differences under about 0.2 between two 24-code figures should be
read as noise. Solid directions are steadier: about 31 per code with a spread of 3 between codes.
Script: `analysis_scripts/variance_noise.py`.

The run wanders instead of improving: variance between 0.80 and 1.34, anchors between 0.73 and 0.82, and the
final checkpoint has 17 defective codes of 128 (the trainer's last probe read 0.62 clean). The 7,500
checkpoint is the healthiest of the continuation and is no better than J4 at 5,000. The memory fault was
active during the last hour of this run (torch's libraries were among the damaged files), so an effect on
the training itself cannot be ruled out.

**HARDWARE ALERT, 22:20 CST 13 Nov: the memory fault is now breaking programs.** Within about an hour the
in-memory copies of at least six files went bad (all clean on disk, all with the same signature: top bit
flipped in bytes 3 and 7 of one 8-byte word): Python's `logging` module, two of numpy's library files, a
torch or CUDA library (not yet located), the checkpoint saved at update 6,000, and other programs' libraries
(Firefox and another desktop app are crashing repeatedly). Consequences:
- **No evaluation can be run right now.** A new Python process crashes as soon as it imports torch. The run
  that is training loaded its libraries earlier and is still going, but it shares the damaged numpy pages
  and could crash or miscompute.
- **Results on disk are intact as far as checked**: checkpoints at 7,500 (J6) and J4's final one, and the
  96-anchor set, read the same from memory and disk.
- **A restart clears the damaged in-memory copies** (the disk copies are clean) but does not fix the memory,
  so it will happen again. The owner should run a memory test (memtest86+) and replace or exclude the bad
  memory before trusting further runs.
- Tools: `analysis_scripts/direct_read_plain.py [--evict] <file>` compares (and can drop) a file's in-memory
  copy using the system Python, which still works.

**Finished 01:27 CST 13 Nov: `A1b_d64_follow_capped_even_code_leak05` (leaky activation, 1,000 updates).**
Against I1 (same recipe without the leak) at 1,000 updates:
- Dead units in the last hidden layer: 9 (I1: 99). Units in use per code there: 328 of 512.
- Total variance 0.645 (I1: 0.501); solid directions 33 (32); above 0.01: 53 (52); effective 11.6 (10.9).
- Fresh codes clean 128 of 128 (127); anchors reproduced 0.82 (0.82).
- Field errors unchanged (consistency 0.039 against 0.037, eikonal 0.0098 against 0.0093); thin-wall sites
  about 107 per pass (87).
Reading: the leak stops units dying and costs nothing visible. The direction count is the same at this point,
which is expected: only a fifth of the units had died by 1,000 updates in I1; the damage came later (305 dead
at 4,000). A longer run is needed to see whether the leak prevents the later stall.

**Finished about 01:55 CST 13 Nov: `N1_d64_follow_capped_even_code_layernorm_last` (layer norm on the last
hidden layer only, 1,000 updates).** It did not fix unit death: 88 dead units in the last hidden layer (I1
without it: 99; A1b with the leak: 9). The dead units are held off by the code's own scale and shift, which
are applied after the norm (their multiplier averages 0.40 and their shift -0.63). Otherwise it is close to
I1: total variance 0.599, 32 solid directions, 128 of 128 fresh codes clean, anchors 0.81 (worst 0.52),
consistency 0.035, eikonal 0.0089.

**Crashed at start-up, about 01:57:** `N2_d64_follow_capped_even_code_layernorm_all` (layer norm on all 8
layers) ran out of GPU memory in its first update. No result.

**Finished about 02:20 CST 13 Nov: `K1_d64_anchors_only_capacity`** (the decoder trained on nothing but the
64 Thingi anchors of the third set, 1,500 updates). Anchor overlap levelled off at about 0.84 (0.83 at 1,000,
0.86 at 1,250, 0.84 at 1,500), not the 0.95 that would have shown ample room. Per anchor: 2 at or above
0.95, 25 between 0.90 and 0.95, 21 between 0.80 and 0.90, 13 between 0.50 and 0.80, 3 below 0.50. The larger
half of the objects averages 0.91, the smaller (thin) half 0.76, and two thin rods vanish entirely (overlap
0.00). Fine detail is rounded (spiky ball, gear teeth). 88 units of the last hidden layer are dead even here.
Reading: not a clean answer on size. The misses are thin objects, which a whole-ball field loss barely
notices, and fine detail, which this smooth network rounds off. The full run H3 scores 0.83 on the same
anchors with no anchor below 0.63. Picture: `figures/data_free_anchor_fit_K1_anchors_only_1500.png`.

**Finished: `K2_d64_anchors_only_beta100`** (anchors-only test with activation sharpness 100). A small gain,
not the detail: anchors reproduced 0.86 (K1: 0.84), median 0.91 (0.89), larger objects 0.92 (0.91), thin ones
0.79 (0.76), 33 anchors at or above 0.90 (27), 64 dead units in the last hidden layer (88). In the picture
the toothed plate still has no teeth, the spiky ball is still a bumpy ball, edges are still rounded, and
three thin rods are still lost. Picture: `figures/data_free_anchor_fit_K2_anchors_only_beta100_1500.png`.

**Training (owner's request; started 13:20 CST 13 Nov; about 2 h 20 min; nothing else queued):**
`J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k`, 5,000 updates from scratch. It is the
combined main recipe J2 (even-code penalty, no filled centre, leaky activation 0.05, size-weight ceiling 500)
with three changes: the new 96-anchor set with thin shapes
(`datasets/data_free_anchors_procedural_thick_plus_thin_20261113_v6`: the 64 thick anchors plus 16 thin rods
and 16 thin sheets), with 6 anchors per pass in place of 4 so each anchor is practised as often as before;
activation sharpness 100 in place of 20 (`softplus_beta=100`); and three Fourier bands on the coordinates
(`fourier_bands=3`). A first launch with the 64 thick anchors
(`J3_d64_nocentre_procedural_even_code_leak05_beta100_fourier3_5k`) was stopped at update 35 at the owner's
request and has no result. Extra things to watch here: whether thin rod anchors turn into strings of beads
(as in the earlier run P3) and the worst anchor score. Unlike the anchors-only
tests, all topology terms are on. Questions: does it stay valid with a network that ripples more easily, do
edges and detail get crisper, and does the field get worse? Compare with J2 at the same updates (1,000: total
variance 0.565, 26 solid directions, 127 of 128 clean, anchors 0.73, 22 dead units; 5,000: 0.786 / 29 / 128 /
0.82 / 93 dead; unseen-object fit 0.787). Watch probes clean, wobble (ripple on the surface; J2 ran at 0.05
to 0.06), thin-wall and gap sites, and consistency and eikonal errors. Checkpoints kept at 1,000, 2,500 and
4,000. Because the anchors and the network both differ from J2, a difference cannot be assigned to one of them.

**J4 finished at 15:50 CST 13 Nov** (96 anchors, sharpness 100, three Fourier bands, 5,000 updates):

| Update | Total variance | Solid directions | Fresh clean / 128 | Anchors (worst) | Size (target 0.156) | Dead units |
|---|---|---|---|---|---|---|
| 1,000 | 0.585 | 21 | 126 | 0.54 (0.00) | 0.141 | 51 |
| 2,500 | 0.804 | 30 | 128 | 0.71 (0.27) | 0.170 | 62 |
| 4,000 | 0.985 | 32 | 120 | 0.76 (0.24) | 0.171 | 71 |
| 5,000 | 1.118 | 31 | 128 | 0.81 (0.48) | 0.172 | 88 |

- Variety kept growing to the end (J2, the same recipe with 64 thick anchors, plain activation and no Fourier
  bands, levelled off near 0.8); directions 31 (J2: 29).
- Validity is less steady: 8 of 128 fresh codes defective at 4,000, none at 5,000; training probes 0.81 to
  1.00 through the run.
- Anchors at 5,000 by group: 64 thick 0.87 (J2's same 64: 0.82); 16 thin rods 0.63, lumpy, 3 in more than one
  piece; 16 thin sheets 0.72, with holes and ragged surfaces in several.
- Field at random codes: slope error 0.103 (0.196 near the surface), consistency miss 0.035; same as J2.
- Six unseen Thingi objects: rod 0.64, gear 0.75, T-bar 0.74, knob 0.87, block with peg 0.84, notched box 0.72.
- Pictures: `figures/data_free_random_shapes_J4_5000.png`, `figures/data_free_anchor_fit_J4_5000.png`,
  `figures/data_free_anchor_fit_thin_J4_5000.png` (the 32 thin anchors),
  `figures/data_free_fit_six_unseen_J4_5000.png`.
- Unseen-object fit, 225 Thingi objects: mean overlap 0.809, the best so far (J2 0.787, H3 0.765). Objects
  with a filled centre 0.882 (J2 0.871); thin or hollow-centred 0.717 (J2 0.679, H3 0.676), 64% of them at or
  above 0.70 (J2 55%, H3 45%); worst object 0.39 (J2 0.21).

**Blind spots of J4 (worst of the 225 unseen fits; picture `figures/data_free_worst_unseen_fits_J4_5000.png`):**
the 16 worst (overlap 0.39 to 0.61) are almost all thin bars, thin plates and open frames, plus a spiky ball
and a block with holes; the best attempts are lumpy, beaded sticks. Across all 225, overlap goes with solidity
(rank correlation +0.79; lowest third 0.69, highest 0.90), volume (+0.74), thickness (+0.71) and being
star-shaped (+0.61), and against rod-like (-0.31) and sheet-like (-0.34) character. Part of this is the
overlap measure itself, which punishes thin shapes more for the same surface error.

**Queued behind J5 (about 25 min):** `V1_d64_follow_capped_even_code_leak05_surface_variety`, 1,000 updates. A
test of a different variety measure: `variety_surface_band=0.1` makes the variety push compare two shapes only
at sample points within 0.1 of either surface, where the field difference equals the distance between the
surfaces. Features then count by surface area, not volume, and field differences far from both surfaces
count for nothing. Otherwise it is exactly the leaky-activation test A1b (filled centre, first Thingi
anchors, even-code penalty, leak 0.05). Compare with A1b at 1,000 updates: total variance 0.645, 33 solid
directions, 128 of 128 clean, anchors 0.82, consistency 0.039, eikonal 0.0098. The logged nearest-neighbour
distance is on a different scale in this run, so do not compare it with other runs' logs.

**Unseen fits re-scored by surface distance (thickness-neutral; same 225 objects, same best codes):**
- Mean distance between the object's surface and the best attempt's: J4 0.0213, J2 0.0242, H3 0.0278 (a
  typical object is about 1 across). F-score at 0.04 (share of surface within 0.04): 0.87, 0.84, 0.79; at
  0.02: 0.62, 0.58, 0.50. Same ranking as by overlap. Attempts in one piece: 223, 225 and 223 of 225.
- The thin-shape gap shrinks but stays: for J4 the thinnest third has surface distance 0.026 against 0.014 for
  the thickest (overlap 0.72 against 0.91). Solidity is still the strongest predictor (rank correlation 0.80);
  volume matters much less than overlap suggested (0.40, was 0.74).
- The worst-objects list changes: by surface distance the worst are mostly blocky objects with features (a
  spiky ball, cubes with holes, puzzle cubes, a notched box, figurines), whose corners, spikes and holes are
  missed by 0.15 to 0.22, plus a few thin rods. Picture: `figures/data_free_worst_unseen_by_surface_J4_5000.png`.
- Per-object results: `analysis_scripts/followup/fit_<run>_surface.json`; script `analysis_scripts/rescore_fits.py`.

**Training (owner's request; started 20:30 CST 13 Nov; about 2 h 20 min; nothing else queued):**
`J6_J4_continued_to_10k`. It continues the best model so far, J4 (96 anchors, even-code penalty, no filled
centre, leak 0.05, sharpness 100, three Fourier bands), from update 5,000 to 10,000 with no setting changed.
J4 was the only even-code run whose variety was still growing at 5,000. Questions: does total variance keep
rising (J4: 0.80 at 2,500, 0.99 at 4,000, 1.12 at 5,000), do the thin anchors improve (thin rods 0.63 with 3
in pieces, thin sheets 0.72 at 5,000), and does validity hold (J4 had 8 defective codes at 4,000 and none at
5,000)? Also watch the size measure (0.172 against the target 0.156 at 5,000) and dead units (88).
Checkpoints kept at 6,000, 7,500 and 9,000.

**Finished: `V1_d64_follow_capped_even_code_leak05_surface_variety`** (the variety push measured by surface
distance within 0.1 of either surface; 1,000 updates). Against A1b, the same recipe with the whole-ball
measure: no difference in variety. Total variance 0.647 (A1b 0.645), 33 solid directions (33), 128 of 128
fresh codes clean (128), anchors 0.81 (0.82), 13 dead units (9). The field is not better: at random codes
the slope error is 0.097 (0.087) and the consistency miss 0.034 (0.030). Thin-wall sites are about half
(62 per pass against 107) and gap sites too (28 against 57). Random shapes look the same kind of lobed blob.
Picture: `figures/data_free_random_shapes_V1_1000.png`.

**J5 finished about 19:00 CST 13 Nov and is worse than J4 on every measure:**

| Update | Total variance | Solid directions | Fresh clean / 128 | Anchors (worst) |
|---|---|---|---|---|
| 1,000 | 1.036 | 24 | 87 | 0.36 (0.00) |
| 2,500 | 0.804 | 27 | 120 | 0.55 (0.02) |
| 4,000 | 0.725 | 25 | 128 | 0.61 (0.09) |
| 5,000 | 1.023 | 28 | 109 | 0.67 (0.24) |

- Validity: 19 of 128 fresh codes defective at 5,000 (J4: 0), 41 at 1,000.
- Anchors at 5,000: 64 thick 0.74 (J4 0.87); 16 thin rods 0.43 with 11 in more than one piece (J4 0.63 and
  3); 16 thin sheets 0.58 (J4 0.72).
- Field: no better. At anchor codes the slope error is 0.078 (J4 0.066) and the field is 0.024 from the
  object's true distance (J4 0.013), although eikonal and consistency now act there. Random codes the same
  as J4.
- Thin-wall sites about 300 per pass over the last 500 updates (J4 272), so removing the thicken term changed
  little there. Dead units 154 (J4 88). Variance sits in fewer directions (effective 5.7, J4 8.0).
- Three things changed together (no thicken term, field terms on anchors, a third of the anchor practice), so
  the cause cannot be pinned down; the drop in anchor fit is what less practice would do.
- Pictures: `figures/data_free_random_shapes_J5_5000.png`, `figures/data_free_anchor_fit_J5_5000.png`,
  `figures/data_free_anchor_fit_thin_J5_5000.png`.

**What J5 was:** `J5_d64_nocentre_procedural96_nothicken_anchorfield_5k`, 5,000 updates from scratch. The J4 recipe (96
anchors, even-code penalty, no filled centre, leak 0.05, sharpness 100, three Fourier bands) with three
changes:
- **no thicken term**: its weight is fixed at zero (`mult_ranges.thicken=0,0`); thin-wall sites are still
  counted in the log, and the logged thicken multiplier will read 0;
- **eikonal and consistency on the anchor codes too** (`anchor_exempt_field=False`); until now anchors were
  left out of both;
- **8 anchor samples per update** (`anchor_codes_per_pass=2`, four passes per update) in place of 24, so
  about 94% of the codes in an update are random. Each anchor is then practised a third as often as in J4.
  This reading of "8 at a time" is an assumption; the other reading is 8 per pass (32 per update).
Watch: thin-wall sites (expected to rise with thicken gone) and probes clean; whether thin rod anchors stop
breaking into pieces; anchors reproduced (may fall with less practice); and the consistency and eikonal
errors, which now include the anchor codes and so are not directly comparable with earlier runs' logs.
Three things change at once, so a difference from J4 cannot be assigned to one of them.

**J4 so far (checkpoints at 1,000 and 2,500 evaluated at 15:35 CST; final 5,000 expected about 15:55):**
- 1,000: total variance 0.585, 21 solid directions, 126 of 128 fresh codes clean, anchors 0.54 (worst 0.00),
  51 dead units. 2,500: 0.804, 30 solid (54 above 0.01), 128 of 128 clean, anchors 0.71 (worst 0.27), size
  0.170, 62 dead units. J2 at 2,500 for comparison: 0.704 / 29 / 128 / anchors 0.80 (64 thick anchors) / 51.
- Anchors by group at 2,500: the 64 thick ones 0.78 (none below 0.59); the 16 thin rods 0.49, with 10 below
  0.50 and 5 decoded in more than one piece; the 16 thin sheets 0.63.
- Surface ripple settled: wobble 0.27 before training, 0.05 to 0.07 from update 500 on.
- Training probes less steady than J2's: clean 0.94 at 1,000, 0.81 at 1,500, 0.88 at 2,000 and 3,000, 1.00
  at 2,500 and 3,500, 0.94 at 3,750. Anchors in training reached 0.80 by 3,750 and are still rising.
- Field at 2,500, random codes: slope error 0.103 (0.165 near the surface), consistency miss 0.032; about
  the same as J2.

**Finished: the two Fourier tests** (anchors-only test with sines and cosines of the coordinates added):
- `F1_d64_anchors_only_fourier3` (3 bands): chunky objects fit better and edges are crisper, thin objects
  worse. Mean 0.84 (K1: 0.84), median 0.92 (0.89), 46 anchors at or above 0.90 (27), larger objects 0.93
  (0.91), thin ones 0.75 (0.76), 5 anchors below 0.50 (3), four of them lost entirely. Dead units in the last
  hidden layer: 277 (K1: 88).
- `F2_d64_anchors_only_fourier5` (5 bands): worse. Mean 0.78, median 0.84, 9 anchors at or above 0.90, rippled
  surfaces, and thin objects come out as fragments and specks. Dead units: 400 of 512.
- Reading: three bands help overall shape and edges of chunky objects but not fine teeth, and they triple
  unit death; five bands bring the ripple and specks the design note warned about.
- Pictures: `figures/data_free_anchor_fit_F1_anchors_only_fourier3_1500.png`,
  `figures/data_free_anchor_fit_F2_anchors_only_fourier5_1500.png`.

**What K2 was:** `K2_d64_anchors_only_beta100`, 1,500
updates. It repeats the anchors-only test K1 (decoder trained on nothing but the 64 Thingi anchors of the third
set) with one change: the activation's sharpness is 100 in place of 20 (`softplus_beta=100`), which lets the
network draw sharper creases. Question: do the rounded details come back (the toothed plate's teeth, the spiky
ball, square edges)? If yes, sharp-edged anchors become worth building; if not, the network itself is the
limit. Compare with K1: anchors reproduced 0.84 (larger objects 0.91, thin ones 0.76, two thin rods lost),
88 dead units in the last hidden layer. Read `anchors iou` in the probe lines; the picture is drawn afterwards.

**Queued behind it (owner's request: "the Fourier fix"), each 1,500 updates, about 25 min:**
1. `F1_d64_anchors_only_fourier3`: the same anchors-only test with three Fourier bands added to the coordinates
   the main network sees (`fourier_bands=3`: sines and cosines with periods 2, 1 and 0.5), activation
   sharpness back at the usual 20.
2. `F2_d64_anchors_only_fourier5`: the same with five bands (periods down to 0.125, about the size of a gear
   tooth).
Same question as K2: do the rounded details come back, and at what cost in ripple? Judge by `anchors iou`
against K1's 0.84 and by the anchor pictures. These test fitting only; they say nothing yet about validity
or variety in a full run.

**New anchor set built (owner's request, 13 Nov), not yet used in a run:**
`datasets/data_free_anchors_procedural_thick_plus_thin_20261113_v6`, 96 geometric anchors: the 64 thick ones
of the earlier set unchanged (numbers 0 to 63), plus 16 thin rods (64 to 79, 0.10 to 0.18 across) and 16 thin
sheets (80 to 95, 0.09 to 0.14 thick). The thick ones are 0.16 to 0.68 across. Picture of all 96:
`figures/data_free_procedural_anchors_thick_plus_thin_v6.png`. Things to know before training on it: the
trainer's minimum feature is 0.08, so the thinnest additions are close to what its thin-wall term pushes
against; an earlier run whose anchors were mostly this thin (P3) turned rods into strings of beads after
about 1,250 updates, though here only a third of the set is thin; and with 96 anchors each one is practised
less often unless `anchor_codes_per_pass` is raised from 4 to 6.

**Unseen-object fit on J2 at 5,000 (225 Thingi objects none of the models trained on):** mean overlap 0.787
(H3 at 7,000: 0.765; H2 at 2,500: 0.770), the best so far. Objects with a filled centre (126): 0.871 (H3
0.835). Thin or hollow-centred objects (99): 0.679 (H3 0.676), with 55% reaching 0.70 (H3 45%) but a lower
worst case (0.21 against 0.36). On the 83 filled-centre objects also unseen by the older models: J2 0.888, the
32-dimension anchor model L2 0.876, H3 0.855, no anchors 0.789. This is the first model to beat L2 there.

**Finished: `J2_d64_nocentre_procedural_even_code_leak05_5k`** (geometric anchors, even-code penalty, no
filled centre, leaky activation 0.05; 5,000 updates from scratch):

| Update | Total variance | Solid directions | Fresh clean / 128 | Anchors (worst) | Size (target 0.156) | Dead units, last hidden layer |
|---|---|---|---|---|---|---|
| 1,000 | 0.565 | 26 | 127 | 0.73 (0.19) | 0.157 | 22 |
| 2,500 | 0.704 | 29 | 128 | 0.80 (0.60) | 0.159 | 51 |
| 4,000 | 0.885 | 30 | 124 | 0.82 (0.58) | 0.174 | 69 |
| 5,000 | 0.786 | 29 | 128 | 0.82 (0.57) | 0.170 | 93 |

- Valid at the end (128 of 128) with 29 to 30 solid directions and 63 of 64 above 0.005.
- Variety levelled off after about 2,500 updates (0.70 to 0.89), as it did in I2 without the leak. So unit
  death was not the main cause of that levelling off.
- The leak slowed unit death (93 at 5,000; without it 305 at 4,000 in I2) but did not stop it.
- At 1,000 updates it is slightly behind J1, the same recipe without the leak (J1: 0.725 / 29 / anchors 0.78).
- Six unseen Thingi objects: rod 0.52, gear 0.77, T-bar 0.64, knob 0.88, block with peg 0.84, notched box
  0.73. Chunky objects as good as any model so far; the thin rod is worse than in H3 (0.89), whose anchors
  included thin rods (these geometric anchors are all thick).
- Pictures: `figures/data_free_random_shapes_J2_5000.png`, `figures/data_free_anchor_fit_J2_5000.png`,
  `figures/data_free_fit_six_unseen_J2_5000.png`.

**What J2 was:** `J2_d64_nocentre_procedural_even_code_leak05_5k`, described two paragraphs down. It stayed on
the leaky activation, because layer norm did not work (next paragraph).

**Finished about 02:44: `N2c_d64_follow_capped_even_code_layernorm_all_p1024`** (layer norm on all 8 hidden
layers, 1,024 points per pass, 1,000 updates). It did not fix unit death: 142 dead units in the last hidden
layer (no fix: 99; last-layer layer norm: 88; leaky activation: 9). Total variance 0.499, 29 solid
directions, 128 of 128 fresh codes clean, anchors 0.80, consistency 0.033, eikonal 0.0097. As in N1 the dead
units are held off by the code's own scale and shift, applied after the norm (multiplier 0.44, shift -0.69).
Layers 1 to 7 are healthy. It ran at half the usual points per pass, so it is not like-for-like.

**What N2c was:** `N2c_d64_follow_capped_even_code_layernorm_all_p1024`, the all-layers
layer-norm test again with 1,024 points per pass in place of 2,048 (owner's choice, to stay well inside GPU
memory; 1,000 updates). That makes it not quite like-for-like with I1, A1b and N1. (An intermediate version
with 1,536 points, N2b, was replaced before it started and never ran.)

**Queued last (owner's request; starts by itself when N2c ends; about 2 h 15 min):**
`J2_d64_nocentre_procedural_even_code_leak05_5k`, 5,000 updates from scratch. It is the stopped J1 run again
(geometric anchors, even-code penalty, no filled centre, size-weight ceiling 500) with the leaky activation
(`activation_leak=0.05`). The owner's condition: if the all-layers layer-norm test N2c works well, use layer
norm in this run instead of the leak. So when N2c ends its dead-unit count and other numbers are checked; if
layer norm is the better fix, J2 is stopped within its first minutes and relaunched with layer norm on all
layers and 1,024 points per pass.
Compare with J1 at 1,000 updates (total variance 0.725, 29 solid directions, 126 of 128 clean, anchors 0.78,
116 dead units). Checkpoints kept at 1,000, 2,500 and 4,000.

**What A1b was:** `A1b_d64_follow_capped_even_code_leak05`,
1,000 updates. Exactly the I1 recipe (filled centre, first Thingi anchors, capped-follow, even-code penalty at
weight 100) with one change, `activation_leak=0.05`: the main network's activation keeps a slope of 0.05 when
a unit is off, so a unit pushed far negative can come back. It tests the dead-unit finding further down.
Compare with I1 at 1,000 updates: 99 dead units in the last hidden layer, 32 solid directions, total variance
0.501, 127 of 128 fresh codes clean, anchors 0.82. (A first attempt, `A1_d64_follow_capped_even_code_leak05`,
crashed at start-up from GPU memory; the activation code was fixed and it was relaunched as A1b.)

**Queued, in this order (owner's request: layer norm as a remedy for dying units):**
1. `N1_d64_follow_capped_even_code_layernorm_last` (1,000 updates, about 25 min): the I1 recipe with layer norm
   on the last hidden layer only (`layer_norm_layers=1`), the layer where units die. No leak.
2. `N2_d64_follow_capped_even_code_layernorm_all` (1,000 updates, about 25 to 35 min): the same with layer norm
   on all 8 hidden layers (`layer_norm_layers=8`). It uses more GPU memory and might fail at start-up.
3. `K1_d64_anchors_only_capacity` (1,500 updates): the anchors-only capacity test described further down.
All three are judged against I1 on dead units (count with `analysis_scripts/dead_units_why.py`), solid
directions, total variance, fresh codes clean and anchors reproduced.

**Stopped at the owner's request, 00:57 CST 13 Nov:** `J1_d64_nocentre_procedural_even_code_5k` at update
1,195 of 5,000 (checkpoints at 1,000 and 1,100 kept). Its description and 1,000-update result follow.
It combined the three ingredients that
had not been run together:
- **geometric (procedural) anchors**, the thick set `datasets/data_free_anchors_procedural_thick_20261112_v5`
  (rods, sheets, bowls, plates with posts, balls with limbs, lumps), with `anchor_exempt_size=True` as in P4;
- **the even-code penalty** (`code_isotropy_weight=100`), as in I1 and I2;
- **no filled centre**, as in H2, H3 and P4.
Otherwise the capped-follow recipe (64 dimensions, consistency weight capped at 20), with the size-weight
ceiling at 500. It runs on the merged code (the fork's `procedural-anchors` branch merged into the main
data-free branch, commit 73044d8) and is detached, so it survives a session ending.
Compare with: P4 at 2,500 (same anchors, no penalty: total variance 1.231, 20 solid directions, 126 of 128
clean, anchors 0.82; its size-weight ceiling was 50, so its size may be inflated), I2 (penalty, Thingi
anchors, filled centre: 36 solid directions, variance 0.67 to 0.87) and H3. Watch probes clean, `iso/term`
(lower is more even), thin-wall sites, and whether rod anchors break into beads as they did in P3.
Checkpoints kept at 1,000, 2,500 and 4,000.

**J1 at update 1,000:** total variance 0.725, 29 solid directions (48 above
0.01, 63 above 0.005), effective directions 7.7; 126 of 128 fresh codes clean; anchors reproduced 0.78 (worst
0.49); size 0.160 against the target 0.156; 116 dead units in the last hidden layer. For comparison at 1,000
updates: I1 (penalty, Thingi anchors, filled centre) 0.501 / 32 / 127 clean / anchors 0.82 / 99 dead; S4 (no
penalty) 0.762 / 21 / 127 clean / 0.82.

**What K1 is (now third in the queue above):** `K1_d64_anchors_only_capacity`, 1,500 updates. A
capacity test: the same decoder trained on nothing but reproducing the 64 Thingi anchors of the third set
(variety push off, graph and pose terms off, anchors fitted exactly at their codes, 16 anchors per pass).
If anchor overlap climbs well above the 0.75 to 0.84 the full runs reach (say to 0.95), the network is large
enough and the full runs' anchor fit is limited by the competing terms and by training time. If it stalls
near 0.85, size or smoothness of the network is the limit. Read `anchors iou` in the probe lines.

**Earlier plan for the leak test (superseded by the entry at the top):** it is exactly
the I1 recipe (filled centre, first Thingi anchors, capped-follow, even-code penalty at weight 100) with one
change, `activation_leak=0.05`: the main network's activation keeps a slope of 0.05 when a unit is off, so a
unit pushed far negative can come back. It tests the dead-unit finding below. Compare with I1 at 1,000
updates: 99 dead units in the last hidden layer, 32 solid directions, total variance 0.501, 127 of 128 clean,
anchors 0.82. Dead units are counted afterwards with `analysis_scripts/dead_units_why.py`.

**Finding, about 00:50 CST 13 Nov: most of the last hidden layer is dead.** The main network's last hidden layer
(layer 8 of 8, 512 units) has units that are switched off for essentially every code and point: 99 dead at
update 1,000 (I1), 165 at 2,500 (G2), 305 at 4,000 (I2), 405 at 7,000 (H3); a fresh network has about 36.
Layers 1 to 7 are healthy (at most 7% dead). Units in use per code in layer 8: 134 in I2, 83 in H3. The dead
units' inputs sit around -1.7, where the near-ReLU activation passes no gradient, so they cannot come back.
Separately, 96 to 99% of a variety-style gradient lands in the final 512-to-1 layer, whose weights are tiny
(about 0.003 to 0.005). Not yet tested: whether keeping these units alive raises the number of directions.
Scripts: `analysis_scripts/unit_activity.py`, `dead_units_why.py`, `grad_by_layer.py`.

**Before that, nothing was training (as of 23:45 CST 12 Nov).**

**`I2_even_code_5k` finished at 23:08 CST 12 Nov (even-code penalty, continued from 1,000 to 5,000):**

| Update | Total variance | Solid directions | Above 0.005 | Fresh clean / 128 | Anchors | Size (target 0.156) |
|---|---|---|---|---|---|---|
| 1,000 | 0.501 | 32 | 63 | 127 | 0.82 | 0.152 |
| 2,500 | 0.777 | 36 | 63 | 128 | 0.87 | 0.180 |
| 3,500 | 0.854 | 35 | 63 | 127 | 0.88 | 0.172 |
| 4,000 | 0.673 | 33 | 63 | 127 | 0.87 | 0.156 |
| 5,000 | 0.874 | 36 | 63 | 120 | 0.88 | 0.152 |

- The extra directions held for the whole run (33 to 36 solid; runs without the penalty have 17 to 22).
- Total variance stopped growing after about 2,500 (0.67 to 0.87).
- The final checkpoint is a bad moment: 8 of 128 fresh codes defective, and in the last updates the
  consistency error jumped from about 0.038 to 0.062 and eikonal from 0.006 to 0.023. The update-4,000
  checkpoint (`checkpoint_4000.pt`) is the healthier one to use: 127 of 128 clean.
- Thin-wall sites 70 to 390 per pass with the thicken multiplier at its ceiling of 1,000 most of the time.
- At 5,000 every code direction still reaches the main network equally (weakest 0.89 of the strongest); at
  the surface the weakest is 0.10 of the strongest (without the penalty 0.03 to 0.05).
- Pictures: `figures/data_free_random_shapes_I2_5000.png`, `figures/data_free_anchor_fit_I2_5000.png`,
  `figures/data_free_code_path_gains_I2_5000.png`.

**Six unseen Thingi objects fitted on I2 at 4,000 (picture `figures/data_free_fit_six_unseen_I2_4000.png`):**
overlap 0.62 (thin rod, comes out as a string of beads), 0.75 (gear, a smooth lump), 0.61 (T-shaped bar),
0.89 (knob), 0.86 (block with a peg), 0.73 (notched box). On the same objects the no-filled-centre model H3
scored 0.89, 0.73, not scored (its own anchor), 0.79, 0.84, 0.67. So the even-code model is slightly better
on the chunky objects and much worse on the thin rod, which its filled-centre rule does not allow.

**Earlier interruption:** `I2_even_code_5k` stopped at update 2,968 at 20:15 CST 12 Nov, when the session
that launched it ended (the machine itself kept running and was restarted in an orderly way at 21:05; no crash
in the system log). It was resumed from its update-2,900 checkpoint at about 22:20 CST and now runs detached
from any session; about 50 min to reach 5,000. Its checkpoints read identically from memory and disk.
**At update 2,500:** total variance 0.777, 36 solid directions (58 above 0.01, 63 above 0.005), effective
directions 12.9, 128 of 128 fresh codes clean, anchors reproduced 0.87, size measure 0.180 against the target
0.156. The run without the penalty at 2,500 (S5b) had 22 solid directions, 127 clean, anchors 0.86, and total
variance 1.10 at a much larger size (0.287); scaled to the target size that is about 0.60 against 0.67 here.

**No waves along code directions (I2 at 2,500 against S5b at 2,500; picture
`figures/data_free_loop_waves_I2_vs_S5b.png`):** on 11 loops around the code sphere, the shape changes once
per trip. Two shapes are most different when their codes are opposite (180 degrees), with no return on the
way. With the evenness penalty 89% of the field's change is one cycle per trip, 7% two cycles, 1% four or
more; without it 80%, 13% and 2%. So the penalty did not buy its extra directions with wiggles.

**What I2 was (finished; results at the top):**
`I2_even_code_5k` continues the I1 model below from update 1,000 to 5,000, same settings (even-code penalty at
weight 100), with one change: the ceiling on the size term's weight is raised from 50 to 500, because earlier
runs of this recipe grew past the size target after about 1,500 updates with the ceiling at 50. Questions:
does total variance catch up with the runs without the penalty while the extra directions stay (I1 at 1,000:
32 solid directions, total variance 0.501; without the penalty S4 had 21 and 0.762 at 1,000, and S5b 22 and
1.10 at 2,500 at an inflated size), and does validity hold. Watch `iso/term` (0.004 at update 1,000; lower is
more even), probes clean, thin-wall sites and the thicken multiplier. Checkpoints kept at 2,500 and 3,500.

**Finished about 19:15 CST 12 Nov: `I1_d64_follow_capped_even_code`** (1,000 updates). It is the S4 recipe
(filled centre, first Thingi anchor set, capped-follow) with one addition, `code_isotropy_weight=100`: a
penalty that asks every code direction to change the scales and shifts handed to the main network by the same
amount. It acts only on the small side network that turns the code into those scales and shifts.
Result against S4 at the same 1,000 updates:
- More directions in use: 32 solid directions (S4: 21), 52 above 0.01 (31), 63 above 0.005 (44); effective
  directions 10.9 (8.2).
- Less total variance: 0.501 (S4: 0.762). The strongest directions are weaker, the weak ones stronger.
- Validity and anchors unchanged: 127 of 128 fresh codes clean (127), anchors reproduced 0.82 (0.82).
- The penalty did its own job fully: the weakest direction now changes the scales and shifts 0.89 as much as
  the strongest (S4: 0.15). The main network still fades them: at the surface the weakest is 0.08 of the
  strongest (S4: 0.03). `iso/term` fell from 0.024 to 0.004.
- Pictures: `figures/data_free_code_path_gains_I1_1000.png` (and `_S4_1000.png` for comparison),
  `figures/data_free_random_shapes_I1_1000.png`, `figures/data_free_anchor_fit_I1_1000.png`.
One run and one seed; earlier seed repeats at 1,000 updates differed by about one solid direction.

**Best result so far:** `S4_d64_follow_capped` at update 1,000 (surface-following push, fixed-surface
consistency, consistency's weight capped at 20). Total variance 0.762 (the ordinary run G2 had 0.165 at
update 1,000 and 0.327 at 2,500), 21 solid directions (G2: 14), 127 of 128 fresh codes clean, anchors
reproduced 0.82. Costs: consistency error about 0.033 (four times its tolerance), which keeps the graph in
its weaker "sign" mode; about 95 thin-wall and 50 gap sites per pass.
`E1_d64_eikonal_only` (consistency switched off) reached 0.502 total variance, 21 solid directions and 128 of
128 clean at update 1,000, with a rougher field still (consistency error 0.044).

**Finished 01:34 CST 12 Nov:** `S5b_follow_capped_long` (the S4 model continued from update 1,000 to 2,500).
Validity held: 127 of 128 fresh codes clean, probes clean 1.00, anchors reproduced 0.86. Total variance 1.104,
22 solid directions (32 above 0.01, 46 above 0.005). Caveat: the shapes grew. Their size measure is 0.287
against a target of 0.156 (S4 was 0.163), about a third larger in length, and the size multiplier is at its
ceiling of 50. Variance scales with size squared, so the rise from 0.762 is growth, not new variety; scaled
back to S4's size it is about 0.63. Thin-wall sites stayed near 75 per pass with the thicken multiplier at its
ceiling of 1,000. Pictures: `figures/data_free_random_shapes_S5b_2500.png`,
`figures/data_free_anchor_fit_S5b_2500.png`.

**Direction loops on S5b (pictures `figures/data_free_direction_loops_S5b_2500_page1.png` to `_page4.png`,
and `_all.png`):** one trip around the code sphere along each of the 64 code directions, 16 shapes each,
from one base code. Directions 1 to about 20 give clearly different shapes (surface motion 0.36 down to
0.09); directions 49 to 64 (0.03 to 0.02) all decode to nearly the same neutral shape. Measured as one shared
set of directions over 24 codes, the top 12 hold about 80% of the variance and the top 20 about 92%. A few
pure strong-direction codes show loose fragments or holes (directions 1, 3, 13).

**Finished about 02:45 CST 12 Nov:** `H2_d64_nocentre_follow_capped` (S4 recipe without the filled-centre
rule, third anchor set of thin rods, plates, hooks, L and T shapes; 2,500 updates). All 128 fresh codes clean.
Total variance 1.235, 21 solid directions (32 above 0.01, 45 above 0.005). Anchors reproduced 0.75 (worst
0.40): plates, hooks and L shapes come out recognisable, thin rods come out short and lumpy, a spiky ball
comes out as a plain ball. Random shapes are visibly flatter and thinner than S5b's. Same caveat as S5b: the
size measure is 0.252 against a target of 0.156, so part of the variance is size. Thin-wall sites about 113
per pass with the thicken multiplier at its ceiling. Pictures: `figures/data_free_random_shapes_H2_2500.png`,
`figures/data_free_anchor_fit_H2_2500.png`.

**What R1 was (finished; result further down):** `R1_d64_field_ramp500` (1,500 updates from scratch):
the ordinary recipe (G2 settings: 64 dimensions, tolerance 0.008, frozen anchor codes, ordinary consistency)
with the eikonal and consistency weight limits ramped in. Each limit rises geometrically from 0.01 to its
normal 200 over the first 500 updates (about 1.4 at update 250), so variety can develop first. Questions: does
the variety that develops survive once the weights are back, and does the field error come back down?
Compare with G2 at update 1,000 (total variance 0.165, 14 solid directions, consistency error about 0.008)
and S4 (0.762, 21, error 0.033). Watch `mult/consistency` and `mult/eikonal` climbing, the consistency error
after update 500, and live directions. Checkpoints kept at 500 and 1,000.

**Tidy-up test, three attempts done (a short fine-tune of S4 to repair the field without reshaping):**
- `T3_tidy_300_surface_pin` (finished 02:57 CST 12 Nov; surface itself pinned with weight 1,000, only eikonal
  and consistency pushing, 300 updates at learning rate 3e-4): in between the other two and not a repair.
  Consistency error 0.036 to 0.030, flat from about update 1,080 on with its weight at the ceiling of 200;
  eikonal 0.0089 to 0.0082; graph still in "sign" mode. Shapes moved a little: before/after overlap 0.92
  (worst code 0.80), pinned points off by about 0.005.
- `T1_tidy_100` (hold weight 100, full learning rate): field repaired (consistency error 0.036 to 0.011,
  graph back in full mode) but only by flattening the shapes: before/after overlap of the solids 0.52,
  anchors reproduced fell from 0.81 to 0.58.
- `T2_tidy_100_strong_hold` (hold weight 3,000, learning rate 1e-4): shapes held (overlap 0.99) but the
  field barely improved (consistency error 0.036 to 0.034, eikonal unchanged, graph still in "sign" mode).
- Reading: with the shapes held, these field terms do not repair the field in a few hundred updates; when
  the field is repaired, it is by reshaping. The field and the shapes share the same weights.

**R1 finished 03:34 CST 12 Nov: the ramp gave no gain.** Total variance 0.114 at update 500, 0.160 at 1,000,
0.123 at 1,500; solid directions 13, 14, 14; 128 of 128 fresh codes clean throughout; anchors reproduced 0.77,
0.80, 0.84. That matches the ordinary run G2 at 1,000 (0.165, 14). The field error did come back to its
tolerance (0.043 at update 375, 0.008 by about 1,050). Variety did not develop faster while the two weights
were near zero, so in the ordinary recipe their early weight is not what holds variety back.

**P1 failed and was stopped at update 1,125 (13:35 CST 12 Nov).** `P1_d64_nocentre_extent_procedural` was the
first run with the extent-based pose rule. At update 1,000, 63 of 128 random codes decoded to an EMPTY shape
and the rest were 5 to 17% too small; probes clean 0.31, anchors reproduced 0.49, consistency error 0.16 with
the graph terms switched off. A flaw in the new size term as first written was found and fixed (it barely
pulled a shape that was too small back), but that was not the main cause: see P2. Do not use P1's numbers or
checkpoints.

**P2 also failed and was stopped at update 1,025 (14:05 CST 12 Nov).** `P2_d64_nocentre_extent_procedural` was
P1 with the corrected size term: 14 of 128 random codes empty at update 500, 73 of 128 at update 1,000. Do not
use its numbers or checkpoints.

**Finding the cause (forked session). Two short tests, 1,000 updates each:**
- `X1_d64_nocentre_massrule_procedural` (finished 14:32 CST): the procedural anchors under the OLD mass-based
  pose rule, H2's settings otherwise. Clean: 0 empty shapes at update 700, probes clean 1.00 throughout,
  consistency error near 0.04, anchors reproduced 0.57 at update 1,000 and rising. So the procedural anchors
  are not the cause.
- `X2_d64_nocentre_extent_ceiling50` (ended 14:45 CST at update 504): the new pose rule and the new anchors
  with the size weight's ceiling at its normal 50. It failed the same way: at update 500, 32 of 128 random
  codes empty and the rest about 23% too small, consistency error 0.14. So the ceiling of 500 was not the
  cause; the new pose rule itself is. (The run then crashed out of GPU memory at update 504, a separate
  matter.)
- A measurement on the failing models: the anchor pull is the force that shrinks the random shapes' volume
  most, 5 to 10 times harder than in X1. Under the new rule random shapes hold about 1 to 2% of the box by
  volume against 3.5% in X1 and 8% in H2. Why the new rule leads there is not established yet.
- `X3_d64_nocentre_extent_trajectory` (finished about 15:01 CST, 400 updates): X2's first 400 updates again
  with a checkpoint every 100. Random shapes were already at 0.69 of the target size by update 100 (volume
  1.3% of the box; the starting ball has 6.5%) and stayed near 0.78 from update 200 on.
- **Conclusion: the extent-based pose rule does not train with this recipe, and has been set aside.** Under it
  "size" means reach, and a shape that is too small can only grow by pushing thin parts further out, which the
  decoder does last. Under the mass rule it fattens, which is easy (X1 also shrank early and recovered). The
  stuck size term then disturbs the field near the far tips, the consistency error passes the level at which
  the graph terms switch off, and shapes fragment and empty. `pose_power` is back to 2 by default on the
  `procedural-anchors` branch; 6 remains an option.

**P3 was stopped at update 1,550 (15:47 CST 12 Nov): the first procedural anchors were too thin.**
`P3_d64_nocentre_procedural_fit` (mass rule, anchors `..._fit_20261112_v4`) was clean to update 1,250, then at
1,500: probes clean 0.94, 4 of 128 random codes empty, median volume of random shapes down to 0.75% of the
box, and the thin rod anchors decoded as strings of separate beads. Three quarters of those anchors were
under 0.16 across (the Thingi anchors that worked: 7 of 64). So X1's clean 1,000 updates did not show the
anchors were fine; the damage is only slower under the mass rule. Do not use P3's numbers.

**P4 finished about 17:07 CST 12 Nov and is valid (forked session). Nothing from the fork is training or
queued now; the GPU is free.** `P4_d64_nocentre_procedural_thick` at update 2,500:
- total variance 1.231, 20 solid directions (30 above 0.01, 43 above 0.005), 126 of 128 fresh codes clean,
  anchors reproduced 0.82 (worst 0.45), no empty shapes at any check. H2 (Thingi anchors) at 2,500: 1.235, 21,
  128 of 128, 0.75. So about the same on these measures, with the procedural anchors reproduced better.
- size inflated like H2: size measure 0.254 against the target 0.156 (the size weight's ceiling was 50).
- **The anchors' geometry did not spread to random codes.** Measured with the geometric descriptors on 48
  random shapes: all blob-like and star-shaped, the same as H2's random shapes. Of the 64 anchors themselves
  41% are not star-shaped, 19% rod-like, 25% sheet-like.
- **It holds only near each anchor's code:** at cosine 0.9 from an anchor code 36% of shapes are not
  star-shaped (27% rod-like, 22% sheet-like); at 0.7, 14%; at 0.5 and below none. A random code in 64
  dimensions sits at cosine about 0.1 from every anchor.
- Pictures: `figures/data_free_random_shapes_P4_2500.png`, `figures/data_free_anchor_fit_P4_2500.png`.

**What P4 was:** the H2 recipe with the old mass-based pose
rule and THICKER procedural anchors: 64 shapes in 47 families in
`datasets/data_free_anchors_procedural_thick_20261112_v5` (picture
`figures/data_free_procedural_anchors_thick_v5.png`). None is under 0.16 across and they hold 1.1 to 4.5% of
the box by volume, the same range as the Thingi anchors. Six far-reaching anchors (two clubs, ball on a stick,
paddle, two spoons) are scaled down to fit inside the outer radius, so they are smaller than the size target
on purpose; `anchor_exempt_size=True` leaves anchor rows out of the size term so the two do not fight.
Compare with H2 at 2,500 (Thingi anchors: total variance 1.235 at an inflated size, 21 solid directions, 128
of 128 clean, anchors 0.75). Healthy signs: `probes clean` at 1.00 throughout, consistency error near 0.04,
`anchors reproduced` rising past 0.6. Warning signs (seen in P3): probes clean dropping below 1.00, thin-wall
sites above about 500 per pass, anchors reproduced stuck near 0.5. Checkpoints kept at 1,000 and 1,500.

**What P1 and P2 were (both failed; kept for reference):** the H2 recipe (64 dimensions, no filled centre,
capped-follow, consistency cap 20, size-weight ceiling 500) with two changes:
- **Procedural anchors chosen by kind of geometry** instead of Thingi objects: 64 shapes in 48 families (27 rods
  such as L, C, S, fold-back, helix, Y, T; 17 sheets such as plates, bowl, saddle, twisted ribbon; 11 plate or
  ball with rods; 9 lumps). File `datasets/data_free_anchors_procedural_extent6_20261112_v3`, picture
  `figures/data_free_procedural_anchors_extent6_v3.png`.
- **A new pose rule, `pose_power=6`:** where a shape sits and how big it is are measured by its extent (6th power
  of the distance from the origin) instead of by its mass. A ball with a long rod now fits inside the outer
  radius (the ball comes out smaller). The log's `pose c` and `pose s` are the new centre and size errors, in
  the same units as before; the probe readout has new `pose/extent_*` numbers, and the old `pose/trace_*`
  numbers no longer have to match the target.
Two things changed at once, so a difference from H2 cannot be assigned to one of them. Compare with H2 at
2,500 (total variance 1.235 at an inflated size, 21 solid directions, 128 of 128 clean, anchors 0.75). Code is
on branch `procedural-anchors` (worktree `.claude/worktrees/procedural-anchors`); evaluate its checkpoints with
that code, not with the older worktree.

**Unseen-object fit test (run 11:39 to about 13:00 CST 12 Nov; finished):** for each Thingi object the model
never trained on, a code is found by gradient descent and the overlap with the object is scored.
- H3 at 7,000 against H2 at 2,500, same 225 unseen objects: mean overlap 0.765 against 0.770. No gain from the
  extra 4,500 updates, although the direction-spectrum variance doubled. Objects with a filled centre (126):
  0.835 against 0.841. Thin or hollow-centred objects (99): 0.676 against 0.678, and only 45% reach 0.70.
- Against earlier models, on the 83 filled-centre objects unseen by all: H3 0.855, H2 0.860, the 32-dimension
  filled-centre anchor model L2 0.876, the no-anchor model 0.789. L2's anchors came from that same
  filled-centre pool, which favours it.
- Picture of five unseen objects on H3: `figures/data_free_fit_six_unseen_H3_7000.png` (rod 0.89, gear 0.73,
  knob 0.79, block with peg 0.84, notched box 0.67; fine detail such as gear teeth is lost).
- Reading: removing the filled-centre rule made thin shapes reachable (the rod was 0.47 on S4), but training
  longer did not widen what the model can reproduce.

**H3 finished about 05:28 CST 12 Nov (`H3_nocentre_follow_capped_7k`, final at update 7,000):** total variance
1.449, 17 solid directions (24 above 0.01, 34 above 0.005), effective directions 7.4; 126 of 128 fresh codes
clean; anchors reproduced 0.84 (worst 0.62), visibly better than at 2,500 (rods full length, funnel with its
stem); size measure 0.173 against the target 0.156; consistency error about 0.04; thin-wall sites 240 to 630
per pass with the thicken multiplier often at its ceiling. Reading: variety and anchor fit kept improving,
while the number of directions, validity and thin walls slowly got worse. Pictures:
`figures/data_free_random_shapes_H3_7000.png`, `figures/data_free_anchor_fit_H3_7000.png`.

**H3 on the way (checkpoints at 3,500 and 5,000):**
- Size is fixed: the size measure is 0.159 at 3,500 and 0.154 at 5,000 against the target 0.156 (H2 ended at
  0.252). The size weight sits at its new ceiling of 500.
- Total variance 0.825 at 3,500 and 1.120 at 5,000. H2's 1.235 was at the inflated size; scaled to the target
  size it was about 0.76, so variety is genuinely growing.
- Solid directions 19 then 18 (H2: 21); above 0.005: 39 then 35 (H2: 45). Part of that fall is the smaller
  size (the thresholds are absolute); effective directions 8.9 then 7.8, so variance is concentrating in the
  top directions.
- Fresh codes clean 128 then 127 of 128. Anchors reproduced 0.79 then 0.81 (worst 0.29 at 5,000).
- Thin-wall sites rose from about 113 per pass in H2 to roughly 150 to 490; gap sites also up.

**What H3 was (finished; results above):**

1. `H3_nocentre_follow_capped_7k` (started 03:35 CST 12 Nov; owner's request): the long no-filled-centre anchor
   run. It continues H2 (capped-follow recipe, third anchor set) from update 2,500 to 7,000, about 2 hours.
   The capped-follow recipe was chosen because the ramp showed no gain. One setting changed: the ceiling on
   the size term's weight is raised from 50 to 500, because in H2 and S5b the shapes grew about a third past
   the size target with that weight stuck at 50. Expect the size measure (`term/pose_scale`) to fall and the
   shapes to shrink back toward the target early on; watch thin-wall sites and probes clean while that
   happens. Checkpoints kept at 3,500 and 5,000.

**Redundancy test (12 Nov, evening; P4 at 2,500 and H3 at 7,000):** 16 gradient descents on the code from
random starts toward one target. For the model's own shapes all 16 starts recover the shape (overlap 0.94 to
0.99) but end at different codes (cosine to the best code 0.24 to 0.79, median about 0.5 to 0.6), and the
straight path between two such codes keeps the shape (overlap dips only to about 0.8 to 0.9). So many codes
give one shape, joined by nearly flat valleys: the redundancy is unused directions, not separate minima. For
unseen Thingi objects the fits are poor (0.44 to 0.71) and for one (a cube with holes) different starts end
at different shapes. Pictures: `figures/data_free_redundancy_paths_P4_2500.png` and `..._H3_7000.png`.

**Where code directions are lost (H3 at 7,000; picture `figures/data_free_code_path_gains_H3_7000.png`):**
the code goes 64 to 256 to 256 to the 8,192 scales and shifts of the main network. The first layer's weights
are spread almost as evenly as a fresh layer's (singular values 2.59 down to 0.80, fresh 1.71 to 0.60), and
its principal directions rank the same as surface motion (rank correlation 0.97). No stage cuts directions
off; the weakest direction is 0.45 of the strongest after the first layer, 0.14 after the side network and
0.05 at the surface. Not evidence that the network is too small; that needs a wider-network run.

**Stopped or crashed earlier (kept for reference):** `S5_follow_capped_long` (stopped after a few updates to
make room for T2; S5b replaces it), `S2_d64_fixed_surface` and `S3_d64_fixed_surface_follow` (consistency
weight up to 200: shapes grew then fell back to spheres), `D1_d64_nocentre_dial03` (same pattern, then out of
GPU memory; the memory problem is fixed), `G5_d32_unpaired32_tol008_frozen` (no longer needed).

**Measured on G2:** the variety push and the consistency term pull the decoder's weights in almost exactly
opposite directions (cosine -0.95); eikonal the same (-0.97). 98% of both gradients sits in the decoder's
last layer, which sets how far shapes may depart from a sphere.

**Fitting test on S4 (three random unseen Thingi objects):** overlap 0.47 (thin rod, broke into pieces),
0.72 (gear), 0.60 (T-bar). Two of the three are thinner than this model's filled-centre rule allows.
Picture: `figures/data_free_fit_three_random_thingi.png`.

**How wrong the field is (slice z = 0, three random codes, against the true distance to the decoder's own
surface):** S4 mean error 0.013 to 0.020 over the slice and 0.005 to 0.008 within 0.1 of the surface;
ordinary model G2 0.008 to 0.012 and about 0.003. Worst spots about 0.1 in both, far from the surface or on
the shape's inner skeleton. Pictures: `figures/data_free_field_error_capped_follow.png` and
`figures/data_free_field_error_ordinary.png`.

**Hardware warning, confirmed 12 Nov 03:05:** the computer's memory is faulty. Two files are clean on disk
but their copies held in memory each have the top bit flipped in two bytes four apart
(`prepared_v2/37284.json` and `G2_d64_tol008_frozen/metrics.jsonl`; reading G2's metrics the normal way
fails at byte 2,767,059). The same signature both times points at one bad memory cell or line. The main
checkpoints (S4, S5b, H2, G2 at 2,500, T3) and both anchor sets read identically from memory and from disk.
Check any file with `analysis_scripts/direct_read.py <file>`. The owner should run a memory test
(memtest86+) and, until then, treat a surprising result or a file that will not parse as possibly this.

**Earlier note (11 Nov 23:05):** one Thingi metadata file (`prepared_v2/37284.json`, unchanged on disk
since 2 Nov) now reads with two flipped bits. It read correctly earlier that day. Together with two earlier
crashes this points at faulty memory. Checkpoints and results could be affected silently.

**Decisions waiting on the owner:** whether to change the variety measure so it only sees the surface
(soft inside/outside difference); whether to try gradient surgery in the layers before the last one.

## How runs are organised

- Run folders: `/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/<name>/` with
  `train.log` (one line per 25 updates plus a `diag` line per periodic check), `metrics.jsonl`
  (every update), `diagnostics.jsonl` (every check), `config.json`, `checkpoint_last.pt` (every 100 updates).
- One training at a time on the single GPU (RTX 5080, also drives the display), always through
  `scripts/gpu_capped.py`. About 1.4 s per update when alone; slower when an evaluation or the anchor build
  shares the GPU.
- A queued run is a waiting shell script (`queue_run2.sh` or `cont_run.sh`) that starts when the named run's
  trainer exits.
- After-run measurements are JSON files in `analysis_scripts/followup/<tag>.json` (written by `evalrun.py`);
  `python3 analysis_scripts/followup/table.py <tag> ...` prints them side by side (needs the project's
  `.venv/bin/python` only for evalrun, not for table).
- Pictures are in `/home/eric-luo/Downloads/pointcloud-motion-planning/figures/` (`data_free_*.png`).
- The machine has crashed twice without a logged cause. A run whose log has not advanced for several minutes
  while nothing else is using the GPU has probably died; say so.

## Reading the numbers

Health of a running job (from the status script):

| Number | Meaning | Healthy | Worry |
|---|---|---|---|
| skipped updates | updates dropped because the loss was not finite | 0 | any |
| probes clean | share of test shapes that are one piece with no cavity and no handle | 1.00 | below 0.9 twice running |
| consistency / tolerance / multiplier | average miss when stepping from a point to the surface; the multiplier is the weight that holds it at the tolerance | value near tolerance, multiplier moving between about 20 and 200 | value far above tolerance with the multiplier stuck at 200 |
| eikonal / multiplier | error in the field's slope | value under 0.01, multiplier 1 | multiplier rising above 1 (it has started to bind) |
| thin-wall sites per pass | places thinner than the minimum feature size | under about 10 | above 20 and rising; thicken multiplier above 50 |
| gradient size | size of the raw update before clipping | tens | thousands for a long stretch |
| anchors reproduced | overlap (0 to 1) between each real object and what the decoder makes at its code; average and worst | rising to 0.85 or more by update 2,500 | flat near its starting 0.51 |
| live directions | logged count of code directions that move the surface; depends on the field's current error, so treat as a rough guide | 10 to 15 in anchor runs | falling |
| nearest-neighbour distance | how different each shape is from its most similar neighbour | rising over the run (0.03 to 0.05) | flat near zero |

Quality of a finished model (from `evalrun.py`, the numbers that decide between runs):

- **Total variance**: overall amount of shape variety at random codes. Higher is better.
- **Solid directions**: code directions that move the surface by more than 0.02 per unit step. Higher is better.
  Counts above 0.01 and 0.005 show the weaker tail.
- **Effective directions**: how evenly the variety is spread over directions. Noisy (about ±0.6 between seeds).
- **Fresh codes clean (of 128)**: valid single solids among 128 new random codes, confirmed on finer grids.
- **Anchor overlap**: as above, average and worst.
- Seed-to-seed noise at update 1,000: about ±4% in variance and one solid direction. Differences smaller than
  that are not real.

## Results so far (64 anchors unless noted; single seed unless noted)

| Run | What | Updates | Total variance | Solid dirs | Fresh clean / 128 | Anchor overlap |
|---|---|---|---|---|---|---|
| `SH3_SH1_recipe_width768_5k` | SH1 recipe with main width 768: variety same as SH1 (nearest 1 - overlap 0.424 vs 0.428), more accurate field (slope error near surface 0.080 vs 0.097), eikonal multiplier lower (39 vs 45), topology 128/128, anchors 0.87; extra width used, went into accuracy | 5,000 | - | - | 128 | 0.87 (run check) |
| `SH1c_SH1_cont_eikcap16_from6700_release8k_to10k` | SH1 continued to 10,000, eikonal multiplier capped at 16 for 6,700-8,000 then released: at 8,000 nearest 1 - overlap 0.462, solidity 0.597; after tightening at 10,000 back to 0.405 / 0.671; field accuracy restored (slope error near surface 0.101); topology 128/128 | 10,000 | - | - | 128 | 0.88 (run check) |
| `SH2c_SH2_continued_to_5k` | SH2 continued to 5,000 (as S6): fresh topology 128/128 (S6 90/128); variety recovered but behind S6 (nearest 1 - overlap 0.375 vs 0.465) and J4 (0.475); thicker, less concave (solidity 0.729) | 5,000 | - | - | 128 | - |
| `SH2_S5_recipe_no_sphere_base_shells2048_2500` | S5 recipe (no fixed sphere) with 2,048 shared shell points: fresh topology 126/128 (S5 75/128) but shapes collapse to one template with a small knob; nearest 1 - overlap 0.297 (S5 0.371), thicker, less concave | 2,500 | - | - | 126 | 0.59 (run check) |
| `SH1_J4_scratch_shells2048_shared_volw_5k` | J4 from scratch with 2,048 shared shell points for every term (pose/volume integrals volume-weighted): less varied than J4 (nearest 1 - overlap 0.428 vs 0.475), smaller, thinner, less concave (solidity 0.668 vs 0.607), smoother; fresh topology 126/128 (J4 128/128); anchors equal | 5,000 | - | - | 126 | 0.80 |
| `S10_S9_continued_to_20k` | S9 continued from 12,000 to 20,000 at the low learning rate; unseen-object fit level with J7 (0.816 / 0.0206), more accurate field at the surface with little facing dependence, more varied than J7, validity lower and swinging (99 to 118 clean); one anchor code developed a field spike | 20,000 | 1.42 (96 codes) | 25 | 118 | 0.88 |
| `S9_S8_lower_lr_to_12k` | S8 (distilled base-free decoder) continued from 10,000 to 12,000 at a tenth of the learning rate; unseen-object fit recovered to J4 at 5,000's level (overlap 0.808, surface distance 0.0216; J7 at 11,000: 0.815 / 0.0207), more varied than J7, thinner shapes, slightly fewer clean codes | 12,000 | 1.60 (96 codes) | 27 | 117 | 0.86 |
| `S8_distilled_no_base_continued_to_10k` | Base-free decoder started as a distilled copy of J4 at 5,000 (`D0_distill_J4_5000_no_base`), trained on to 10,000 with no base; keeps J4's health and per-layer use of the code, slightly less varied than J6 (J4 continued: 1.42 / 32 / 111 / 0.82), much more accurate slope at the surface (0.21 against 0.45), thinner shapes | 10,000 | 1.32 (96 codes) | 27 | 120 | 0.74 |
| `S7_S6_continued_to_10k` | No-base decoder continued from 5,000 to 10,000; variance and anchors kept rising (variance 1.36 against J6's 1.42, pairwise variety level with J6), but validity swings (64 to 113 clean between checkpoints), solid directions stay at 20 (J6 32) and the first main layer stays at about 14 code directions | 10,000 | 1.36 (96 codes) | 20 | 85 | 0.71 |
| `B7_B6_continued_to_5k_budget05` | Other session: learned rounded-box base (B6 form) from scratch to 5,000; the base fades back to one shared rounded cube (variance share 0.16 at 1,000, 0.03 at 5,000); decoded shapes as varied as J4's by pairwise measures, slightly thicker, run checks healthier (clean 1.00, anchors 0.83) | 5,000 | not measured | - | - | 0.83 (run check) |
| `S6_S5_continued_to_5k` | S5 (decoder without the fixed sphere) continued to 5,000; trains, but behind J4 at 5,000 on every model measure (1.09 / 31 / 128 / 0.81), and J4 has as much side-on surface by then; shapes thinner and more open, with fins and webs | 5,000 | 0.77 (96 codes) | 15 | 90 | 0.64 |
| `S5_J4_recipe_sphere_start_no_base_anchor_hold_surface_slope_2500` | Decoder without the fixed sphere: fitted sphere start, small code spread, outer limit by minimum, anchor pull off 500 updates then ramped in over 500, slope term at the surface. First such run that trains properly: field as accurate as J4's, 6.7% of the surface past side-on (J4 0.4%), variety and anchors behind J4 at 2,500 (0.72 / 30 / 128 / 0.71) but rising; 52 codes in pieces or with a through-hole | 2,500 | 0.57 (96 codes) | 12 | 76 | 0.52 |
| `S4_J4_recipe_sphere_start_no_base_outer_min_1k` | S1 with the outer limit changed from a blend into the sphere's field (which pinned the far field) to a smooth minimum with the bounding ball; far field repaired, but all shapes collapse in the first 25 updates (anchor pull) and regrow as rings with a through-hole | 1,000 | 0.05 (96 codes) | 7 | 31 | 0.29 |
| `S3_J4_recipe_sphere_start_usual_spread_anchor_ramp_1k` | S1 with the code's usual starting spread and the anchor pull ramped in over 300 updates; worse: shapes broke up, shrank and drifted, field error higher still, topology terms off 85% of the run | 1,000 | 0.08 (96 codes) | 11 | 5 | 0.10 |
| `S2_J4_recipe_sphere_start_no_base_lr_third_1k` | S1 at one third of the learning rate; almost the same outcome (field error twice J4's, slope weight at its ceiling, shapes lumpy near-spheres) | 1,000 | 0.11 (96 codes) | 9 | 128 | 0.34 |
| `S1_J4_recipe_sphere_start_no_base_1k` | J4 recipe without the fixed sphere in the formula; main network started from a fitted sphere; valid throughout but shapes stay lumpy near-spheres, field error two to three times J4's and swinging (J4 at 1,000: 0.50 / 20 / 126 / 0.54) | 1,000 | 0.14 (96 codes) | 10 | 128 | 0.37 |
| `W1_J4_recipe_weight_norm_1k` | J4 recipe with weight normalisation on the main network (direction and size of each unit's weights learned separately); worse on every measure than J4 at 1,000: the output layer's single size number fell about 60-fold and swung, hidden sizes did not move | 1,000 | 0.53 (96 codes) | 12 | 57 | 0.25 |
| `C3_J4_recipe_crease2_1k` | J4 recipe plus the crease term (flat faces and sharp edges cheaper than bends) at weight 2; no sharpening (flat share 2.3%), variety unchanged, 44 codes with a through-hole | 1,000 | 0.51 (96 codes) | 20 | 84 | 0.54 |
| `C2_J7_11000_crease10_to_12k` | J7 continued from 11,000 at its learning rate plus the crease term at weight 10; limbs and detail erased, shapes became smooth lumps, variety about nine-fold lower (J7 at 12,000: 1.07); field at random codes more accurate | 12,000 | 0.12 (96 codes) | 13 | 127 | 0.85 |
| `C1_J4_recipe_crease10_1k` | J4 recipe plus the crease term at weight 10; every random shape an axis-aligned box with flat faces and sharp edges (flat share 38% against 1.8%), variety a third lower than J4 at 1,000 (0.50) | 1,000 | 0.32 (96 codes) | 9 | 127 | 0.40 |
| `t_batchwide128` | no anchors, 32 dims | 4,500 | 0.287 | 9 | 122 | n/a |
| `L2_r05_exempt_pull100` | anchors, 32 dims, tolerance 0.005 | 5,000 | 0.153 | 13 | 128 | 0.88 |
| `F3_tol008` | L2 continued with tolerance 0.008 | 6,000 | 0.258 | 14 | 127 | 0.88 |
| `F6_dim64` | 64 dims, tolerance 0.005 | 2,500 | 0.175 | 13 | 128 | 0.87 |
| `G1_d64_tol008_free` | 64 dims, tolerance 0.008, anchor codes trainable | 2,500 | 0.218 | 15 | 128 | 0.87 |
| `G2_d64_tol008_frozen` | same, anchor codes frozen (current base) | 2,500 | 0.327 | 16 | 128 | 0.86 |
| `G3_d64_tol008_frozen_long` | G2 continued | 5,000 | 0.410 | 15 | 124 | 0.89 |
| `G3s_settle` | G3 plus 300 updates at one-tenth learning rate | 5,300 | 0.320 | 15 | 128 | 0.89 |
| `S4_d64_follow_capped` | G2 settings + surface-following push, fixed-surface consistency, consistency weight capped at 20 | 1,000 | 0.762 | 21 | 127 | 0.82 |
| `E1_d64_eikonal_only` | G2 settings with consistency switched off | 1,000 | 0.502 | 21 | 128 | (not recorded here) |
| `S5b_follow_capped_long` | S4 continued; shapes grew about a third larger than the size target, which accounts for the variance gain | 2,500 | 1.104 | 22 | 127 | 0.86 |
| `H2_d64_nocentre_follow_capped` | S4 recipe without the filled-centre rule, third anchor set (thin rods, plates, hooks); shapes also grew past the size target | 2,500 | 1.235 | 21 | 128 | 0.75 |
| `R1_d64_field_ramp500` | G2 settings with the eikonal and consistency weight limits ramped in over the first 500 updates; no gain over G2 (0.160 and 14 at update 1,000) | 1,500 | 0.123 | 14 | 128 | 0.84 |
| `I1_d64_follow_capped_even_code` | S4 recipe plus a penalty making every code direction reach the main network equally (`code_isotropy_weight=100`) | 1,000 | 0.501 | 32 | 127 | 0.82 |
| `V1_d64_follow_capped_even_code_leak05_surface_variety` | A1b with the variety push measured by surface distance (`variety_surface_band=0.1`); same variety as A1b, about half the thin-wall sites, field no better | 1,000 | 0.647 | 33 | 128 | 0.81 |
| `A1b_d64_follow_capped_even_code_leak05` | I1 recipe with a leaky activation (`activation_leak=0.05`); 9 dead units in the last hidden layer against 99 in I1 | 1,000 | 0.645 | 33 | 128 | 0.82 |
| `N1_d64_follow_capped_even_code_layernorm_last` | I1 recipe with layer norm on the last hidden layer; 88 dead units there, so it did not fix unit death | 1,000 | 0.599 | 32 | 128 | 0.81 |
| `N2c_d64_follow_capped_even_code_layernorm_all_p1024` | I1 recipe with layer norm on all 8 hidden layers, 1,024 points per pass; 142 dead units in the last hidden layer, so no fix | 1,000 | 0.499 | 29 | 128 | 0.80 |
| `K1_d64_anchors_only_capacity` | Decoder trained on the 64 third-set Thingi anchors only; overlap levels off near 0.84, thin rods lost | 1,500 | n/a | n/a | n/a | 0.84 |
| `TA2s_settle_6k` | TA1 recipe with the surface-band variety measure (`variety_surface_band=0.1`), 5,000 updates plus 1,000 at one-tenth learning rate; thin-wall and gap sites down a third to a half, variance a quarter lower (96 codes), shapes chunkier; unseen-object fit 0.806 | 6,000 | 0.758 | 31 | 126 | 0.86 |
| `TA1s_settle_6k` | J4 recipe with trainable anchor codes, 5,000 updates plus 1,000 at one-tenth learning rate; same as J4 on variety, directions, validity and unseen-object fit (0.809); the settle stage raised anchors from 0.79 to 0.88 (variance on 96 codes) | 6,000 | 1.024 | 31 | 127 | 0.88 |
| `J7_J6_lower_lr_to_12k` | J6 continued from 10,000 at one-tenth learning rate; anchors much better (0.88 at 11,000: thick 0.93, thin rods 0.76, thin sheets 0.82), defective codes down from 17 to 2 to 7 of 128; variance 1.07 on 96 codes at the end, 1.27 at 11,000 | 12,000 | 1.072 | 26 | 122 | 0.84 |
| `J6_J4_continued_to_10k` | J4 continued from 5,000 to 10,000, nothing changed; wanders instead of improving (variance 0.80 to 1.34, 17 defective codes at the end); memory fault active in its last hour | 10,000 | 1.336 | 31 | 111 | 0.82 |
| `J5_d64_nocentre_procedural96_nothicken_anchorfield_5k` | J4 without the thicken term, with eikonal and consistency on the anchor codes, and 8 anchor samples per update in place of 24; worse than J4 throughout: 19 defective codes, thin rods mostly in pieces | 5,000 | 1.023 | 28 | 109 | 0.67 |
| `J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k` | J2 recipe with the 96-anchor set (thick plus thin rods and sheets), activation sharpness 100 and three Fourier bands; variety kept growing; thin rods fit loosely (0.63), 8 defective codes at 4,000 | 5,000 | 1.118 | 31 | 128 | 0.81 |
| `J2_d64_nocentre_procedural_even_code_leak05_5k` | Geometric (procedural thick) anchors + even-code penalty + no filled centre + leaky activation; variety flat after 2,500; 93 dead units at the end | 5,000 | 0.786 | 29 | 128 | 0.82 |
| `I2_even_code_5k` | I1 continued with the size-weight ceiling at 500; directions held at 33 to 36 all the way, variance flat after 2,500; final checkpoint caught a bad moment (use the 4,000 one: 0.673 / 33 / 127) | 5,000 | 0.874 | 36 | 120 | 0.88 |
| `H3_nocentre_follow_capped_7k` | H2 continued, size weight ceiling raised to 500 so shapes stay at the target size (0.825 / 19 at 3,500; 1.120 / 18 at 5,000) | 7,000 | 1.449 | 17 | 126 | 0.84 |
| `P4_d64_nocentre_procedural_thick` | H2 recipe with 64 procedural anchors chosen by kind of geometry (thick version, far-reaching ones scaled down to fit); random shapes stay blob-like, the anchor geometry holds only near the anchor codes | 2,500 | 1.231 | 20 | 126 | 0.82 |

What has been learned:

- Without anchors the decoder ends up using about 10 code directions and the rest go dead; anchors keep more alive.
- 64 code dimensions, a looser consistency tolerance (0.008) and frozen anchor codes each raised variety.
- Running longer keeps raising variety but starts to cost validity (thin walls, a few shapes with handles);
  a short low-learning-rate "settle" stage repairs validity and gives back some variety.
- Things that did not help: pushing unused directions apart in an already-trained model, a "frame" term,
  a very weak anchor pull, a wider anchor neighbourhood.

## If asked something these notes do not cover

Read the run's `config.json` and `train.log`, and the result JSONs. If still unsure, say what is unknown;
do not guess. The main session can be asked through cross-session messaging (ListAgents, then SendMessage
to the session working in the worktree named above); it may be busy and slow to answer.
