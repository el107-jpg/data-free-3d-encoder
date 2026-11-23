"""How much of one update's gradient comes from each part of the loss?

usage: grad_by_term.py <checkpoint or fresh:<checkpoint>> [...]

Runs one real update of the trainer (its 4 passes, its own batches, weights and multipliers) and, inside it,
takes the gradient of every weighted loss term separately. "fresh:<checkpoint>" builds a new, untrained model
with that checkpoint's settings (update 0). Reported per group of terms:
  - size: length of the group's gradient (weight included), and its share of the sum of the lengths;
  - along the update: the group's gradient projected on the whole gradient, as a share of the whole gradient's
    length. These add up to 1; a negative value means the group pulls against the update on balance.
Gradient clipping rescales the whole gradient and leaves these shares unchanged.
"""
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.train import Trainer

GROUPS = [("variety push", ("variety",)), ("slope (eikonal)", ("eikonal",)), ("consistency", ("consistency",)),
          ("size and position", ("pose",)), ("anchor pull", ("anchor",)), ("evenness of the code", ("iso",)),
          ("topology actions", ("bridge", "delete", "fill", "widen", "thicken")),
          ("other (barrier, volume, origin)", ("barrier", "volume", "origin"))]

for item in sys.argv[1:]:
    item, _, forced = item.partition("::")          # <checkpoint>::key=value,key=value changes settings first
    overrides = [o for o in forced.split(",") if o]
    fresh = item.startswith("fresh:")
    path = item[6:] if fresh else item
    if fresh:
        state = torch.load(path, map_location="cpu", weights_only=False)
        trainer = Trainer(DataFreeConfig.from_dict(state["config"]), device="cuda", tensorboard=False)
    else:
        trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False, overrides=overrides)
    cfg, model = trainer.cfg, trainer.model
    params = [p for p in model.parameters()]
    store, values, weights = {}, {}, {}

    def take(name, term, weight):
        if not torch.is_tensor(term) or not term.requires_grad:
            return
        grads = torch.autograd.grad(weight * term, params, retain_graph=True, allow_unused=True)
        flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])
        store[name] = store.get(name, 0) + flat / int(cfg.accumulate)
        values[name] = values.get(name, 0.0) + float(term.detach()) / int(cfg.accumulate)
        weights[name] = float(weight)

    combine = trainer.multipliers.combine

    def patched_combine(objective, constraints):
        take("variety", objective, 1.0)
        for key, value in constraints.items():
            take(key, value, trainer.multipliers.multiplier(key))
        return combine(objective, constraints)

    trainer.multipliers.combine = patched_combine
    anchor_match, code_isotropy = L.anchor_match, L.code_isotropy

    def patched_anchor(*args, **kwargs):
        value, info = anchor_match(*args, **kwargs)
        take("anchor", value, float(cfg.anchor_weight))
        return value, info

    def patched_iso(*args, **kwargs):
        value, ratio = code_isotropy(*args, **kwargs)
        take("iso", value, float(cfg.code_isotropy_weight))
        return value, ratio

    L.anchor_match, L.code_isotropy = patched_anchor, patched_iso
    try:
        record = trainer.step()
    finally:
        L.anchor_match, L.code_isotropy = anchor_match, code_isotropy
    total = sum(store.values())
    whole = float(total.norm())
    rows = []
    for label, keys in GROUPS:
        names = [n for n in store if any(n == k or n.startswith(k + "_") for k in keys)]
        if not names:
            continue
        vec = sum(store[n] for n in names)
        detail = ", ".join(f"{n} {values[n]:.3g} x {weights[n]:.3g}" for n in names)
        rows.append((label, float(vec.norm()), float((vec * total).sum()) / whole ** 2, detail))
    lengths = sum(r[1] for r in rows)
    name = path.split("/")[-2][:44]
    print(f"\n{name} (update {'0, fresh' if fresh else trainer.step_count - 1}); whole gradient {whole:.1f} (logged {record.get('grad_norm', float('nan')):.1f}, "
          f"clipped to {cfg.grad_clip}); field error {record.get('e', float('nan')):.3f}; topology terms: {record.get('graph_mode')}")
    if overrides:
        print(f"  with {', '.join(overrides)}")
    found = {k: round(v, 1) for k, v in record.items() if k.startswith(("graph/", "find/", "sites/", "n_"))}
    print(f"  topology findings per pass: {found}; action weights: " + ", ".join(
        f"{k} {trainer.multipliers.multiplier(k):.0f}" for k in ("bridge", "delete", "fill", "widen", "thicken")))
    print(f"  {'part of the loss':32s} {'size':>8s} {'share':>6s} {'along the update':>17s}   value x weight")
    for label, size, along, detail in sorted(rows, key=lambda r: -r[1]):
        print(f"  {label:32s} {size:8.1f} {size / lengths:6.2f} {along:17.2f}   {detail[:150]}")
    del trainer, store, total
    torch.cuda.empty_cache()
