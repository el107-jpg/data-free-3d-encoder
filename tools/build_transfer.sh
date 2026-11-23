#!/usr/bin/env bash
# Assemble the data-free latent transfer folder. Copy-only: nothing in the sources is changed.
set -euo pipefail
ROOT=/home/eric-luo/Downloads/pointcloud-motion-planning
DEST=$ROOT/data_free_latent_transfer
JOBS=$HOME/.claude/jobs
MEM=$HOME/.claude/projects/-home-eric-luo-Downloads-pointcloud-motion-planning/memory
BASE=dd2d9c0
cd "$ROOT"
mkdir -p "$DEST"/{code/git,notes/claude_memory,notes/docs,runs,session_scratch,figures,data,environment}

# ---------------------------------------------------------------- code snapshots (data-free files of each branch tip)
snap() {
  local name=$1 ref=$2
  local out="$DEST/code/$name"
  rm -rf "$out"; mkdir -p "$out"
  local files=()
  mapfile -d '' files < <(git diff -z --name-only "$BASE" "$ref")
  git archive "$ref" -- arm_planner/__init__.py pyproject.toml "${files[@]}" | tar -x -C "$out"
  git rev-parse "$ref" > "$out/SOURCE_COMMIT.txt"
  echo "snapshot $name: $(find "$out" -type f | wc -l) files at $(git rev-parse --short "$ref")"
}
snap rounded-box-base rounded-box-base
snap j7-shell-surface experiment/j7-shell-surface
snap procedural-anchors procedural-anchors
cp scripts/gpu_capped.py "$DEST/code/gpu_capped.py"

# ---------------------------------------------------------------- full git history of the data-free branches
BRANCHES=(codex/pointcloud-motion-planning worktree-data-free-connectivity-plan procedural-anchors rounded-box-base experiment/j7-slope-diversity experiment/j7-shell-surface)
git bundle create "$DEST/code/git/data_free_branches.bundle" "${BRANCHES[@]}" 2>&1 | tail -2
for b in "${BRANCHES[@]:1}"; do
  # no author line: the logs go into a public repository
  git log --date=iso --stat --format='commit %H%nDate:   %ad%n%n%w(0,4,4)%B' "$BASE..$b" > "$DEST/code/git/commit_log_${b//\//_}.txt"
done
git log --oneline --graph --decorate --date=short --format='%h %ad %d %s' "${BRANCHES[@]}" > "$DEST/code/git/branch_graph.txt"

# ---------------------------------------------------------------- run history (everything except checkpoints)
rsync -a --exclude='*.pt' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pid' \
  runs/data_free_latent/ "$DEST/runs/data_free_latent/"

# ---------------------------------------------------------------- session scratch (analysis scripts and results kept only in job temp folders)
SCRATCH_EXCL=(--exclude='*.pt' --exclude='*.pth' --exclude='*.so' --exclude='*.so.*' --exclude='*.bin'
  --exclude='np_overlay/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/')
# early runs whose records are byte-identical to the copies already under runs/data_free_latent are skipped
DUP=()
for d in "$JOBS"/8dde3c62/tmp/df_train/*/; do
  n=$(basename "$d")
  if [ -d "runs/data_free_latent/$n" ] && diff -rq -x '*.pt' "$d" "runs/data_free_latent/$n" >/dev/null 2>&1; then
    DUP+=(--exclude="/df_train/$n/"); echo "scratch: df_train/$n identical to runs/data_free_latent/$n, skipped"
  fi
done
rsync -a --prune-empty-dirs "${SCRATCH_EXCL[@]}" "${DUP[@]}" "$JOBS/8dde3c62/tmp/" "$DEST/session_scratch/8dde3c62_data-free-latent-space-review/"
rsync -a --prune-empty-dirs "${SCRATCH_EXCL[@]}" "$JOBS/f264ef61/tmp/" "$DEST/session_scratch/f264ef61_decoder-shape-residual-parameterization/"
rsync -a --prune-empty-dirs "${SCRATCH_EXCL[@]}" "$JOBS/ce6bff81/tmp/" "$DEST/session_scratch/ce6bff81_sphere-latent-space-prior-work/"

# ---------------------------------------------------------------- pictures
rsync -a --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='/cleanup_before_after.png' --exclude='/shape_render_check.png' --exclude='/novel_descent_path.png' \
  --exclude='/min_cost_connect.png' --exclude='/sdf_ft_before_after.png' --exclude='/lift_vs_closing.png' \
  --exclude='/priors_gallery.png' --exclude='/sdf_ft_field_slices.png' --exclude='/priors_gallery_large.png' \
  figures/ "$DEST/figures/"

# ---------------------------------------------------------------- notes
cp -p "Data-free latent space.md" "$DEST/notes/"
git show rounded-box-base:docs/data_free_connectivity_plan.md > "$DEST/notes/docs/data_free_connectivity_plan.md"
git show rounded-box-base:docs/data_free_implementation_spec.md > "$DEST/notes/docs/data_free_implementation_spec.md"
cp -p "$MEM"/*.md "$DEST/notes/claude_memory/"
cp -p CLAUDE.md "$DEST/notes/CLAUDE.md"
cp -p "$HOME/.claude/agents/run-monitor.md" "$DEST/notes/run-monitor.agent.md"

# ---------------------------------------------------------------- data: the anchor set the current recipe needs, metadata for the rest
V6=data_free_anchors_procedural_thick_plus_thin_20261113_v6
mkdir -p "$DEST/data/$V6"
cp -p "datasets/$V6"/{anchors.pt,anchors.json,NOTE.txt} "$DEST/data/$V6/"
# anchors.pt is over GitHub's 100 MB file limit; a compressed copy is what the git repository tracks
[ "$DEST/data/$V6/anchors.pt.xz" -nt "$DEST/data/$V6/anchors.pt" ] || xz -6 -T4 -c "$DEST/data/$V6/anchors.pt" > "$DEST/data/$V6/anchors.pt.xz"
for d in datasets/data_free_*; do
  n=$(basename "$d"); [ "$n" = "$V6" ] && continue
  mkdir -p "$DEST/data/other_anchor_sets_metadata_only/$n"
  cp -p "$d/anchors.json" "$DEST/data/other_anchor_sets_metadata_only/$n/"
done

# ---------------------------------------------------------------- environment record
git show rounded-box-base:pyproject.toml > "$DEST/environment/pyproject.toml"
cp -p requirements-source-snapshot.txt "$DEST/environment/" 2>/dev/null || true
uv pip freeze --python .venv/bin/python > "$DEST/environment/pip_freeze.txt" 2>/dev/null || .venv/bin/python -m pip freeze > "$DEST/environment/pip_freeze.txt"
{
  echo "recorded: $(date '+%Y-%m-%d %H:%M %Z')"
  echo "os: $(. /etc/os-release; echo "$PRETTY_NAME"), kernel $(uname -r)"
  echo "python: $(.venv/bin/python -c 'import sys; print(sys.version.split()[0])')"
  .venv/bin/python -c 'import torch, numpy, scipy, skimage; print("torch:", torch.__version__, "cuda build:", torch.version.cuda); print("numpy:", numpy.__version__); print("scipy:", scipy.__version__); print("scikit-image:", skimage.__version__)'
  echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
} > "$DEST/environment/machine.txt"

# ---------------------------------------------------------------- list of checkpoints left behind
{
  printf 'path\tbytes\tmodified\n'
  find runs/data_free_latent -type f -name '*.pt' -printf '%p\t%s\t%TY-%Tm-%Td %TH:%TM\n' | sort
} > "$DEST/CHECKPOINTS_NOT_COPIED.tsv"

sync
echo "== done"; du -sh --exclude=.git "$DEST"; du -sh "$DEST"/*; echo "files: $(find "$DEST" -path "$DEST/.git" -prune -o -type f -print | wc -l)"
echo "pt files in transfer:"; find "$DEST" -path "$DEST/.git" -prune -o -name '*.pt' -printf '%s %p\n'
