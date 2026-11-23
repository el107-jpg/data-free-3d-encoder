"""Cost of Newton landing vs a training update (J4 at 5,000, GPU, under gpu_capped)."""
import time, torch
from arm_planner.data_free.train import Trainer
t = Trainer.from_checkpoint("/home/eric-luo/Downloads/pointcloud-motion-planning/runs/data_free_latent/J4_d64_nocentre_procedural96_even_code_leak05_beta100_fourier3_5k/checkpoint_last.pt", device="cuda", tensorboard=False)
model = t.model
sync = torch.cuda.synchronize
def timed(fn, reps):
    fn(); sync(); a = time.perf_counter()
    for _ in range(reps): fn()
    sync(); return (time.perf_counter() - a) / reps * 1000
t.optimizer.step = lambda *a, **k: None                  # keep the weights fixed while timing
step_ms = timed(lambda: t.step(), 3)
codes = torch.nn.functional.normalize(torch.randn(128, t.cfg.latent_dim, generator=torch.Generator().manual_seed(0)), dim=-1).cuda()
x0 = (torch.rand(128, 2048, 3, generator=torch.Generator().manual_seed(1)) * 2 - 1).cuda()
def land(k):
    for s in range(0, 128, 32):
        mod = model.prepare(codes[s:s + 32]); x = x0[s:s + 32]
        for _ in range(k):
            x = x.detach().requires_grad_(True)
            with torch.enable_grad():
                f = model.query(mod, x)
                (g,) = torch.autograd.grad(f.sum(), x)       # slope w.r.t. the points only, no weight gradients
            x = (x - (f / g.square().sum(-1).clamp_min(1e-8))[..., None] * g).detach()
    return x
res = {k: timed(lambda: land(k), 3) for k in (1, 2, 3)}
print(f"one training update (4 passes x 32 codes x 2,048 points, everything): {step_ms:.0f} ms")
for k, ms in res.items():
    print(f"{k} Newton step(s) for 128 codes x 2,048 points: {ms:.0f} ms = {100 * ms / step_ms:.1f}% of an update")
print("logged breakdown of a J4 update (ms):", {k.split('/')[1]: round(v) for k, v in t.step().items() if k.startswith('time_ms/')})
