#!/usr/bin/env bash
# Rebuild the original working layout on a new device from this folder.
#
#   bash tools/restore_on_new_device.sh /path/to/pointcloud-motion-planning
#
# Creates that folder as a git repository holding every data-free branch with full history, one worktree per
# branch under .claude/worktrees/ (as on the original machine), and copies in the run history, pictures, the anchor
# set and the GPU launcher. It does not create a Python environment and does not touch ~/.claude; see README.md.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=${1:?usage: restore_on_new_device.sh /path/to/pointcloud-motion-planning}
if [ -e "$REPO" ]; then echo "$REPO already exists; give a folder that does not exist yet" >&2; exit 1; fi
MAIN=codex/pointcloud-motion-planning
V6=data_free_anchors_procedural_thick_plus_thin_20261113_v6
if [ ! -f "$HERE/code/git/data_free_branches.bundle" ]; then
  echo "code/git/data_free_branches.bundle is missing. The GitHub copy leaves it out (README.md, 'The GitHub copy')." >&2
  echo "Copy that one file from the original machine into code/git/, or use the code folders as they are (README.md, option B)." >&2
  exit 1
fi
if [ ! -f "$HERE/data/$V6/anchors.pt" ]; then
  echo "unpacking the anchor set"
  xz -dk "$HERE/data/$V6/anchors.pt.xz"
fi

git clone -q -b "$MAIN" "$HERE/code/git/data_free_branches.bundle" "$REPO"
cd "$REPO"
for b in worktree-data-free-connectivity-plan procedural-anchors rounded-box-base experiment/j7-slope-diversity experiment/j7-shell-surface; do
  git branch -q --no-track "$b" "origin/$b"
done
git remote remove origin
git remote add origin https://github.com/el107-jpg/pointcloud-motion-planning.git
for pair in "data-free-connectivity-plan worktree-data-free-connectivity-plan" "procedural-anchors procedural-anchors" \
            "rounded-box-base rounded-box-base" "j7-slope-diversity experiment/j7-slope-diversity" \
            "j7-shell-surface experiment/j7-shell-surface"; do
  set -- $pair
  git worktree add -q ".claude/worktrees/$1" "$2"
done

mkdir -p runs datasets scripts
cp -a "$HERE/runs/data_free_latent" runs/
cp -a "$HERE/figures" figures
cp -a "$HERE/data/$V6" datasets/
cp -a "$HERE/code/gpu_capped.py" scripts/gpu_capped.py
cp -a "$HERE/notes/CLAUDE.md" CLAUDE.md
printf 'figures/\n' >> .git/info/exclude

echo "restored into $REPO"
git branch --format='  %(refname:short) %(objectname:short)'
echo "next: create the Python environment (README.md, 'Python environment'), then run the tests:"
echo "  cd $REPO/.claude/worktrees/rounded-box-base && PYTHONPATH=. python -m pytest tests -q"
