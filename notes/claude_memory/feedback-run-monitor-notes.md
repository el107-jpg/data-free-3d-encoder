---
name: feedback-run-monitor-notes
description: "Keep the run-monitor notes file current whenever a data-free training run is started, queued, stopped or finishes, so the user's phone-reachable monitor agent can report on it"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8dde3c62-9b5e-4c58-883c-36d05f64c978
  modified: 2026-11-12T04:59:31.413Z
---

The user asked (2026-11-11) for a separate agent they can ask about the training runs at any time, including from their phone, and said I should tell it each time I start a new run. The setup: agent definition `~/.claude/agents/run-monitor.md` (Sonnet, read-only), live script `runs/data_free_latent/monitor/status.py`, and notes `runs/data_free_latent/monitor/RUNS.md`.

Whenever a data-free run is started, queued, stopped, finishes, or gets a measured result, rewrite the "Now" section of `RUNS.md` (and add finished runs to its results table): what is training, why, what to compare it with, what is queued, and any decision waiting on the user. Update its "Last edited" line.

**Why:** the monitor only knows what the notes and logs say; stale notes make it tell the user wrong things.
**How to apply:** treat the notes update as part of launching or evaluating a run, not an afterthought. `RUNS.md` lives in the main checkout's `runs/` folder, so write it in a scratch folder and copy it over (the worktree session cannot edit the main checkout with the edit tools). See [[data-free-latent-work]] and [[feedback-images-and-reporting]].
