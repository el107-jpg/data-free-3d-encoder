"""Train the data-free latent space decoder (``arm_planner.data_free.train``).

    PYTHONPATH=. .venv/bin/python scripts/train_data_free_latent.py --output-dir runs/data_free_x \\
        [--steps N] [--device cuda] [--seed S] [--resume] [--set key=value ...]

``--set`` overrides any ``DataFreeConfig`` field (``--set lr=1e-4``,
``--set tolerances.eikonal=0.02``). Writes ``metrics.jsonl``,
``diagnostics.jsonl``, TensorBoard scalars under ``tb/`` and
``checkpoint_last.pt`` into the output directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_planner.data_free.train import main  # noqa: E402

if __name__ == "__main__":
    main()
