"""Current value of the even-code-use term on saved models and on a freshly initialised one.

usage: iso_value.py <checkpoint> [<checkpoint> ...]
"""
import sys

import torch

from arm_planner.data_free import losses as L
from arm_planner.data_free.decoder import ModulatedSphereDecoder
from arm_planner.data_free.train import Trainer

gen = torch.Generator().manual_seed(5)
for path in sys.argv[1:]:
    trainer = Trainer.from_checkpoint(path, device="cuda", tensorboard=False)
    cfg = trainer.cfg
    codes = torch.nn.functional.normalize(torch.randn(64, cfg.latent_dim, generator=gen), dim=-1).cuda()
    term, ratio = L.code_isotropy(trainer.model, codes, 32, gen)
    torch.manual_seed(0)
    fresh = ModulatedSphereDecoder(cfg).cuda()
    term0, ratio0 = L.code_isotropy(fresh, codes, 32, gen)
    print(f"{path.split('/')[-2]} (update {trainer.step_count}): term {float(term):.4f}, typical strong/weak ratio {float(ratio):.2f}; "
          f"freshly initialised: term {float(term0):.4f}, ratio {float(ratio0):.2f}")
