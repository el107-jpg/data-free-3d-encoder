"""Print the ``--set key=value`` arguments that reproduce an earlier run's settings from scratch.

usage: recipe_from_run.py <run folder with config.json> [--check]
Every field of the run's saved config that differs from the trainer's defaults is printed as one
``--set`` argument per line (dict fields entry by entry). ``steps`` and ``seed`` are left out: the
launcher passes them. With --check the arguments are applied to a default config and compared with the
saved one; any field that still differs is reported and the exit code is 1.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from arm_planner.data_free.config import DataFreeConfig  # noqa: E402
from arm_planner.data_free.train import apply_overrides  # noqa: E402

SKIP = {"steps", "seed"}


def text(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(text(v) for v in value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def main() -> None:
    folder = sys.argv[1]
    saved = json.load(open(os.path.join(folder, "config.json")))
    known = DataFreeConfig().to_dict()
    sets = []
    for key, value in saved.items():
        if key in SKIP or key not in known:
            continue
        base = known[key]
        if isinstance(value, dict):
            for name, entry in value.items():
                old = base.get(name) if isinstance(base, dict) else None
                if old is None or list(entry) != list(old) if isinstance(entry, (list, tuple)) else entry != old:
                    sets.append(f"{key}.{name}={text(entry)}")
        elif (list(value) != list(base)) if isinstance(value, (list, tuple)) and base is not None else value != base:
            sets.append(f"{key}={text(value)}")
    for item in sets:
        print("--set")
        print(item)
    if "--check" in sys.argv:
        built = apply_overrides(DataFreeConfig(), sets).to_dict()
        wrong = []
        for key, value in saved.items():
            if key in SKIP or key not in built:
                continue
            a, b = json.loads(json.dumps(built[key])), json.loads(json.dumps(value))
            if a != b:
                wrong.append((key, a, b))
        for key, a, b in wrong:
            print(f"DIFFERS {key}: rebuilt {a} saved {b}", file=sys.stderr)
        print(f"{len(sets)} settings differ from the defaults; rebuilt config matches the saved one: {not wrong}", file=sys.stderr)
        sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
