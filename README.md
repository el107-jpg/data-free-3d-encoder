# Data-free latent space: transfer folder

Everything needed to move the data-free shape-encoding work to another device: the code (all branches, with
history), every note, the full record of every run and ablation, and the pictures. Model checkpoints are not
included.

- Snapshot taken: 2026-11-22, about 23:45 CST.
- Taken from: `/home/eric-luo/Downloads/pointcloud-motion-planning` (Ubuntu 26.04, RTX 5080).
- Size: about 1.1 GB. The originals were only read; nothing was moved or changed.
- This folder is also a git repository, published at <https://github.com/el107-jpg/data-free-3d-encoder>. The
  published copy leaves a few things out; see "The GitHub copy" below.

The project in one paragraph: a decoder-only 3-D shape generator. A network turns a code (a unit vector of 32 or
64 numbers) into a solid shape, with no shape dataset and no encoder. Random codes should give valid, varied
solids, to be used later as obstacles for training the arm motion planner. Training balances a push for variety
against constraints that keep each shape a valid single solid; a few fixed "anchor" objects are reproduced at fixed
codes so that code directions do not go unused.

## What is where

| Folder / file | What it holds | Size |
|---|---|---|
| `code/rounded-box-base/` | Newest code line (commit `52c2223`): the data-free package, trainer, scripts, tests, docs | 1 MB |
| `code/j7-shell-surface/` | The other active line (commit `1cdae25`): distillation, slope and surface variety, background checks | 1 MB |
| `code/procedural-anchors/` | Older line (commit `45a21f9`) kept for its guarded launchers and queue scripts | 1 MB |
| `code/gpu_capped.py` | GPU memory / busy-time launcher (was `scripts/gpu_capped.py`, not in git) | |
| `code/git/data_free_branches.bundle` | Full git history of all five data-free branches plus the main branch | 3 MB |
| `code/git/commit_log_*.txt`, `branch_graph.txt` | Every commit message with the files it changed, readable without git | |
| `notes/` | Design note, plan and spec, Claude's memory notes (the running history), project instructions | 0.1 MB |
| `runs/data_free_latent/` | All 121 run folders without checkpoints, plus `monitor/`, `analysis_scripts/`, `queue_logs/`, `review_probes/` | 611 MB |
| `run_configs_from_checkpoints/` | Exact settings of 115 runs, read out of their last checkpoint | 1 MB |
| `RUN_INDEX.tsv` | One row per run: updates logged, dates, recorded commit, launcher, checkpoints left behind | |
| `session_scratch/` | Analysis scripts and results that only existed in session temp folders; conversation extracts | 117 MB |
| `figures/` | Every data-free picture (368 PNG/JPG), the runs summary page and the blind-spot viewer | 124 MB |
| `data/` | The anchor set the current recipe trains against (`anchors.pt`, plus a compressed copy for git), and descriptions of the other nine | 188 MB |
| `environment/` | Package versions on the original machine | |
| `CHECKPOINTS_NOT_COPIED.tsv` | Every checkpoint left behind, with size and date | |
| `SHA256SUMS` | Checksums of every file here | |
| `tools/` | Restore script for the new device; scripts that built and checked this folder | |

## First thing after copying

    cd data_free_latent_transfer
    sha256sum -c --quiet SHA256SUMS && echo all files intact

(After a `git clone` instead of a copy, use the two commands under "The GitHub copy".)

This matters more than usual: the original machine has faulty memory that damages cached copies of files (history in
`notes/claude_memory/data-free-latent-work.md`). Every file here was compared with its source by reading both
straight from disk, and `SHA256SUMS` was computed the same way, so a clean check means the copy matches the disk.

## Setting up on the new device

### Option A: rebuild the original layout (recommended)

    bash tools/restore_on_new_device.sh /path/to/pointcloud-motion-planning

This creates the repository with every branch and its history, one worktree per branch under
`.claude/worktrees/` (as on the original machine), and copies in `runs/data_free_latent`, `figures`, the anchor
set (`datasets/`), `scripts/gpu_capped.py` and `CLAUDE.md`. `origin` is set to the GitHub address; none of the
data-free branches has ever been pushed, so this folder is their only copy besides the original machine.

Choosing the same path as before (`/home/eric-luo/Downloads/pointcloud-motion-planning`) lets the launchers and
most analysis scripts run unchanged; see "Fixed paths" below.

### Option B: use the code folders as they are

Each `code/<line>/` folder is a self-contained project root:

    cd code/rounded-box-base
    PYTHONPATH=. python -m pytest tests -q

One script needs the rest of the repository (`scripts/build_data_free_anchors.py`, which imports
`arm_planner.object_geometry`); for that, use Option A.

### Python environment

The package needs only `torch`, `numpy`, `scipy`, `scikit-image` and `tensorboard`; scripts and tests also use
`matplotlib`, `trimesh` and `pytest`. The original machine ran Python 3.12.16 with torch 2.7.0+cu128, numpy 1.26.4,
scipy 1.17.1 and scikit-image 0.26.0. `environment/pip_freeze.txt` has every version (use it as a reference, not as
an install file: it contains a local editable install and a CUDA-specific torch build).

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python torch            # pick the build for the new device's GPU
    uv pip install --python .venv/bin/python numpy scipy scikit-image tensorboard matplotlib trimesh pytest

### Check it works

    cd <repo>/.claude/worktrees/rounded-box-base        # or code/rounded-box-base
    PYTHONPATH=. <repo>/.venv/bin/python -m pytest tests -q

Expected, as measured from this folder before transfer (CPU): `rounded-box-base` 327 passed, 2 skipped;
`j7-shell-surface` 268 passed, 2 skipped; `procedural-anchors` 242 passed, 2 skipped.

### Train

A run from scratch with an existing run's recipe (this is how SH3 was started from SH1's settings):

    cd <repo>/.claude/worktrees/rounded-box-base
    PYTHONPATH=. <repo>/.venv/bin/python <repo>/scripts/gpu_capped.py --gpu-fraction 0.75 --gpu-util 0.7 \
      scripts/train_data_free_latent.py --output-dir <repo>/runs/data_free_latent/<new name> --device cuda \
      --config-from <repo>/runs/data_free_latent/SH1_J4_scratch_shells2048_shared_volw_5k/config.json \
      --steps 5000 --seed 0 \
      --set anchor_path=<repo>/datasets/data_free_anchors_procedural_thick_plus_thin_20261113_v6/anchors.pt

- `--set key=value` overrides any setting (`arm_planner/data_free/config.py` lists them); `--resume` continues a run
  from its `checkpoint_last.pt`.
- Every saved `config.json` holds the old absolute `anchor_path`, so pass `--set anchor_path=...` unless the
  repository sits at the same path as before.
- The 0.75 / 0.7 caps exist because the original GPU also drove the desktop. Keep them if that is true of the new
  device; otherwise they can be raised (`--gpu-util 1.0` turns the pauses off).
- Real launchers, including waiting for the GPU, snapshots and resumes, are in `runs/data_free_latent/queue_logs/`.

### Fixed paths

The package, the trainer and the tests contain no machine-specific paths. These do:

- Launchers and analysis scripts (`runs/data_free_latent/queue_logs`, `analysis_scripts`, `review_probes`, about
  half of `session_scratch`) refer to `/home/eric-luo/Downloads/pointcloud-motion-planning/...`.
- Some analysis scripts write to `/home/eric-luo/.claude/jobs/8dde3c62/tmp` or `.../f264ef61/tmp`. Those folders are
  now `session_scratch/8dde3c62_.../` and `session_scratch/f264ef61_.../`.
- `config.json` files hold the absolute `anchor_path` (see above).

Either restore to the same path, or change the prefix in the scripts you use.

### Claude Code on the new device (optional)

- Memory notes: copy `notes/claude_memory/*` to `~/.claude/projects/<project path with every / replaced by ->/memory/`
  (for the same path as before: `~/.claude/projects/-home-eric-luo-Downloads-pointcloud-motion-planning/memory/`).
- Run-monitor agent: copy `notes/run-monitor.agent.md` to `~/.claude/agents/run-monitor.md`.
- `notes/CLAUDE.md` is the project instruction file (GPU caps, pool cache, where figures go); the restore script
  puts it at the repository root.

## Where the history is

Read in this order:

1. `notes/Data-free latent space.md`: the original design note.
2. `notes/docs/data_free_connectivity_plan.md` and `data_free_implementation_spec.md`: the plan and the spec the
   code was built from (identical on every branch).
3. `runs/data_free_latent/monitor/RUNS.md`: the main record. Its "Now" section is a dated log of every run with
   the reason for it, what it was compared with and the measured result; "Results so far" is the table; "Reading
   the numbers" explains each measure.
4. `notes/claude_memory/data-free-latent-work.md`: the condensed running history across sessions, including dead
   ends, lessons, and the faulty-memory episodes. `feedback-*.md` are standing working preferences.
5. `code/git/commit_log_<branch>.txt`: what changed in the code and why, commit by commit.
6. `figures/data_free_runs_summary/data_free_runs.html`: a browsable summary page with pictures.

Per run, in `runs/data_free_latent/<run>/`: `train.log`, `metrics.jsonl` (every update), `diagnostics.jsonl` (every
periodic check), `config.json`, and `tb/` (TensorBoard). Runs that continued another run have no `config.json` and
their logs do not list the changed settings; for those use `run_configs_from_checkpoints/<run>.json`, which is the
full setting list stored in the run's last checkpoint (for the 75 runs that have both, it equals `config.json` in 72;
the other 3 differ only in settings changed on a later resume: step count, check interval, one added switch).

After-run measurements are in `runs/data_free_latent/analysis_scripts/` (scripts plus their JSON results;
`followup/` has the per-model tables, `seed_gap/` the point-sampling study). `session_scratch/` adds what was never
copied there: the first nine runs (`df_train/a` to `f_thin_cap1000`, 9 Nov, before runs were written to `runs/`) and
three other early ones (`h_acc64`, `n_phase2_decaying_aborted`, `s_global_pool128`), dry runs,
capacity / rank / gradient-conflict checks and the sampling study data. `session_scratch/ce6bff81_.../` is the
workspace the runs summary page was built in: it holds plain-text extracts of the working conversations
(`main.txt`, `turns*.txt`, `summaries.txt`, `chunk*.txt`), which are the fullest record of why each thing was tried.

### Run names (orientation only; RUNS.md is the authority)

| Names | When | What they were about |
|---|---|---|
| `a` ... `f` (in `session_scratch`), `g` ... `z` | 9-11 Nov | First experiments without anchors: pose, variety measures, repulsion; `w_anchors64` is the first anchor run |
| `d1`, `d2`, `L1_r05`, `L2_r05`, `F2_settle` ... `F6`, `G*` | 11 Nov | Thingi10K anchors, 32 to 64 dimensions, tolerances, frozen anchor codes |
| `S2_d64` ... `S5b`, `E1`, `H*`, `I*`, `R1`, `T1`-`T3` | 11-12 Nov | Surface-following push, capped consistency, no filled centre, even-code penalty, field ramp, tidy stages |
| `P1`-`P4`, `X1`-`X3`, `D1` | 11-12 Nov | Procedural anchors chosen by kind of geometry; pose and size rules |
| `A1*`, `N*`, `V1`, `K*`, `F1`, `F2_d64` | 13 Nov | Leaky activation, layer norm, surface-band variety, anchors-only capacity |
| `J1`-`J7`, `L1_J4`, `TA*` | 13-14 Nov | The reference recipe (J4) and its continuations; trainable anchor codes with settle stages |
| `C1`-`C3`, `W1` | 14 Nov | Crease term; weight normalisation |
| `S0`, `S1_J4` ... `S10`, `D0` | 14-18 Nov | Decoder without the fixed sphere base; distillation into it |
| `B1`-`B17` | 14-17 Nov | Learned rounded-box base, correction cost, bends and tapers |
| `SD1`, `SD2` | 18-19 Nov | J7's recipe from scratch with slope variety, then shell points and surface scoring |
| `S1x`, `SH1`-`SH3c` | 21-22 Nov | Shell point sampling, wider network, slope-weight cap, variety-weight controller |

Two letters were used twice: `S2`-`S5b` with `d64`/`follow` in the name are the 12 Nov runs; `S1`-`S10` with
`J4_recipe`/`continued` are the no-base runs.

## Code lines and the runs they produced

All five branches start from the main branch at `dd2d9c0` and exist only locally (never pushed).

| Branch | Tip | In `code/` as | Used for |
|---|---|---|---|
| `worktree-data-free-connectivity-plan` | `6e3965e` | (in the bundle; it is an ancestor of `j7-shell-surface`) | The original line: all runs not listed below |
| `procedural-anchors` | `45a21f9` | `procedural-anchors/` | Procedural anchor builder, guarded launchers; recorded for `TA1`, `TA1s`, `TA2`, `TA2s` |
| `rounded-box-base` | `52c2223` | `rounded-box-base/` | `B1`-`B17`, `SH1`-`SH3c` (box base, shells, field caps, variety controller) |
| `experiment/j7-slope-diversity` | `cb51ea4` | (in the bundle; ancestor of `j7-shell-surface`) | `SD1` |
| `experiment/j7-shell-surface` | `1cdae25` | `j7-shell-surface/` | `SD2` (started at `5f40367`) |

The lines have not been merged: `rounded-box-base` does not contain the distillation or slope/surface variety code,
and `j7-shell-surface` does not contain the box base, shells or caps. Where a run recorded its commit, it is in the
`recorded_commit` column of `RUN_INDEX.tsv`; launchers in `queue_logs/` name the commit in their first lines.

## What was left behind

- **Checkpoints: 309 files, 15.5 GB**, listed in `CHECKPOINTS_NOT_COPIED.tsv`. Without them no run can be resumed
  or re-measured on the new device, and no new pictures can be made of old models. If you want a few, each
  `checkpoint_last.pt` is about 51 MB (89 MB for the width-768 runs). The ones the notes lean on most: `J4_...5k`
  (reference recipe), `B12_B11_cont_corrcost50_to9k` (best of the box-base line), `SH3_SH1_recipe_width768_5k` (what
  the queued `SH3c` continues from), `J7_J6_lower_lr_to_12k` and `S10_S9_continued_to_20k`. Copy them into the same
  run folder under `runs/data_free_latent/`.
- **Nine other anchor and candidate sets (2.1 GB)** in `datasets/data_free_*`. Only their descriptions are here
  (`data/other_anchor_sets_metadata_only/`). The procedural ones can be rebuilt with
  `scripts/build_data_free_procedural_anchors.py` (arguments are recorded in each `anchors.json`); the Thingi10K ones
  need the Thingi10K download. Of the 121 runs, 48 use the included set (J4, J5, J6 and every run from 14 Nov on), 55
  use one of these older sets, and 15 use no anchors.
- **Checkpoints and anchor grids in session temp folders** (3.7 GB of `.pt`), and a private copy of numpy made
  while diagnosing the memory fault.
- **The rest of the repository's working state**: the Python environment, the other datasets and runs, and the
  uncommitted latent-set encoder work in the main checkout. The repository's committed code is in the bundle.
- **`worktree-pushout-audit`**: a sixth local branch, about the decoder-object database and SDF fine-tuning, not
  the data-free work. It is not in the bundle.
- **Nine pictures in `figures/`** from that other work (`cleanup_before_after`, `shape_render_check`,
  `novel_descent_path`, `min_cost_connect`, `sdf_ft_before_after`, `sdf_ft_field_slices`, `lift_vs_closing`,
  `priors_gallery`, `priors_gallery_large`).
- **The raw Claude session transcripts** (`~/.claude/projects/-home-eric-luo-Downloads-pointcloud-motion-planning*`).
  Plain-text extracts of them are in `session_scratch/ce6bff81_.../`.

## The GitHub copy

The repository at <https://github.com/el107-jpg/data-free-3d-encoder> is public, while the repository this work
came from is private, so the published copy leaves out what is not about the data-free encoder (`.gitignore` is the
exact list):

- `code/git/data_free_branches.bundle`: it carries the whole private repository, not only the data-free code.
  Without it `tools/restore_on_new_device.sh` cannot run; copy that one 3 MB file across separately, or use the
  code folders directly (option B). The three code folders and the commit logs are published.
- `session_scratch/ce6bff81_.../`: the conversation extracts.
- Six of the nine memory notes (other projects, general working preferences) and `notes/CLAUDE.md`. The three notes
  about this work are published.
- `anchors.pt` itself (170 MB, over GitHub's 100 MB limit). `anchors.pt.xz` (25 MB) is published in its place.
- Two empty folders (git does not store those).

After cloning:

    git clone https://github.com/el107-jpg/data-free-3d-encoder.git && cd data-free-3d-encoder
    xz -dk data/data_free_anchors_procedural_thick_plus_thin_20261113_v6/anchors.pt.xz
    sha256sum -c --quiet --ignore-missing SHA256SUMS && echo all files intact

The commit logs here carry no author names or addresses. Absolute paths from the original machine do appear in
logs, launchers and settings files.

## State when the snapshot was taken

- No data-free run was training. `SH3c_SH3_cont_variety_controller_to10k` was queued and had not started (the GPU was
  held by another project's job), so its folder here holds only the "queued" line. Its launcher is
  `runs/data_free_latent/queue_logs/launch_SH3c_SH3_cont_variety_controller_to10k.sh`.
- All six worktrees were clean: every code change is committed and therefore in the bundle.
- To pick up anything written after the snapshot, run `tools/refresh_on_original_machine.sh` on the original
  machine and copy the folder again.

## How this folder was checked

- All copied files were compared with their sources using direct-from-disk reads; the code and doc files were
  compared with their git commits.
- The bundle was cloned and checked (`git fsck`), and all six branch tips match the original repository.
- The three code folders pass their tests when run from here (counts above).
- `tools/restore_on_new_device.sh` was run into a scratch folder: branches, worktrees, runs, anchors and tests all
  came up as expected.
