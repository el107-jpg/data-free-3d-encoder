"""Compute the trainer's unmodified diagnostics on CPU from an exact saved state."""
import json
from pathlib import Path
import sys
import torch
from arm_planner.data_free.train import Trainer

def main():
    checkpoint, destination = map(Path, sys.argv[1:])
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    runner = Trainer.from_checkpoint(str(checkpoint), device='cpu', tensorboard=False)
    model = runner.model
    query = model.query
    # Preserve point order and batch, limiting temporary pointwise activations in dense passes.
    def bounded_query(mod, x):
        if x.ndim != 3:
            return query(mod, x)
        limit = max(256, 16384 // max(1, x.shape[0]))
        if x.shape[1] <= limit:
            return query(mod, x)
        return torch.cat([query(mod, points) for points in x.split(limit, dim=1)], dim=1)
    model.query = bounded_query
    result = runner.run_diagnostics()
    assert result['step'] == runner.step_count
    temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result))
    temporary.replace(destination)
    runner.close()

if __name__=='__main__':main()
