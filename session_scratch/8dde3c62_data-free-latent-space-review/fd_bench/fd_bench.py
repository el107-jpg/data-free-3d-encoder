"""Finite-difference vs autodiff spatial gradients for UnpackAttentionDecoder (measurement only).

Usage: fd_bench.py --out results.json [--run RUN_DIR --codes codes.pt] [--sections time,acc,pgrad,amp]
"""
import argparse, json, statistics, time
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from arm_planner.object_geometry.latent_set import UnpackAttentionDecoder

MICRO = [32]
BOX = 1.0  # queries uniform in [-BOX, BOX]^3 (sphere-normalized fitting units)
HS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]  # 1e-4 is an extra beyond the requested list
CENTRAL = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float32)
# Standard tetrahedron stencil. NOTE the task text listed the 4th offset as (1,1,-1); that set does not
# satisfy sum(o)=0 and sum(o o^T)=4I, so it does not give the gradient. (1,1,1) is the standard one.
TETRA = torch.tensor([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], dtype=torch.float32)
TETRA_AS_WRITTEN = torch.tensor([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1]], dtype=torch.float32)
FORWARD = torch.eye(3)


def fd_grad(dec, ctx, x, method, h, f0=None, chunk=None):
    """Returns (grad BxQx3, f(x) or None). One prepared context shared by all stencil points."""
    B, Q, _ = x.shape
    off = {'central': CENTRAL, 'tetra': TETRA, 'forward': FORWARD}[method].to(x)
    K = len(off)
    pts = (x[:, :, None, :] + h * off).reshape(B, Q * K, 3)
    need_f0 = method == 'forward' and f0 is None
    if need_f0:
        pts = torch.cat((x, pts), 1)
    if chunk is None:
        vals = dec.query(ctx, pts)
    else:
        vals = torch.cat([dec.query(ctx, pts[:, i:i + chunk]) for i in range(0, pts.shape[1], chunk)], 1)
    if need_f0:
        f0, vals = vals[:, :Q], vals[:, Q:]
    vals = vals.reshape(B, Q, K)
    if method == 'central':
        g = (vals[..., 0::2] - vals[..., 1::2]) / (2 * h)
    elif method == 'tetra':
        g = (vals[..., None] * off).sum(-2) / (4 * h)
    else:
        g = (vals - f0[..., None]) / h
    return g, f0


def ad_grad(dec, ctx, x, create_graph):
    x = x.requires_grad_(True)
    f = dec.query(ctx, x)
    g, = torch.autograd.grad(f.sum(), x, create_graph=create_graph)
    return g, f


def losses(dec, codes, x, method, h, projection):
    ctx = dec.prepare(codes)
    if method == 'autodiff':
        g, f = ad_grad(dec, ctx, x.clone(), True)
    else:
        f = dec.query(ctx, x) if (projection and method != 'forward') else None
        g, f0 = fd_grad(dec, ctx, x, method, h)
        f = f if f is not None else f0
    eik = ((g.norm(dim=-1) - 1) ** 2).mean()
    if not projection:
        return eik, eik.new_zeros(())
    proj = dec.query(ctx, x - f[..., None] * g).abs().mean()
    return eik, proj


def step(dec, codes, x, method, h, projection, amp=None):
    """Autodiff needs the MATH attention kernel: the fused CUDA kernels have no double backward.
    FD uses the default (fused) kernel unless the method name ends with '+math'."""
    if method == 'autodiff' or method.endswith('+math'):
        with sdpa_kernel(SDPBackend.MATH):
            return _step(dec, codes, x, method.replace('+math', ''), h, projection, amp)
    return _step(dec, codes, x, method, h, projection, amp)


def _step(dec, codes, x, method, h, projection, amp=None):
    for p in dec.parameters():
        p.grad = None
    codes.grad = None
    B, te, tp = len(codes), 0., 0.
    for i in range(0, B, MICRO[0]):  # gradient accumulation over code microbatches (repo trainer also microbatches)
        c, xs = codes[i:i + MICRO[0]], x[i:i + MICRO[0]]
        if amp is None:
            eik, proj = losses(dec, c, xs, method, h, projection)
        else:
            with torch.autocast('cuda', dtype=amp):
                eik, proj = losses(dec, c, xs, method, h, projection)
        w = len(c) / B
        ((eik + proj) * w).backward()
        te += float(eik.detach()) * w; tp += float(proj.detach()) * w
    return te, tp


def time_step(dec, codes, x, method, h, projection, iters, amp=None):
    torch.cuda.empty_cache()
    try:
        for _ in range(3):
            step(dec, codes, x, method, h, projection, amp)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize(); t = time.perf_counter()
            vals = step(dec, codes, x, method, h, projection, amp)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        return dict(ms_median=1e3 * statistics.median(times), ms_min=1e3 * min(times), ms_max=1e3 * max(times),
                    peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30, eik=vals[0], proj=vals[1], iters=iters)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' not in str(e).lower() and not isinstance(e, torch.OutOfMemoryError):
            raise
        for p in dec.parameters():
            p.grad = None
        codes.grad = None
        torch.cuda.empty_cache()
        return dict(oom=True)


def vec_stats(g, ref, mask=None):
    if mask is not None:
        g, ref = g[mask], ref[mask]
    g, ref = g.reshape(-1, 3).double(), ref.reshape(-1, 3).double()
    rel = (g - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(g, ref, dim=-1)
    dn = (g.norm(dim=-1) - ref.norm(dim=-1)).abs()
    q = lambda t, p: float(torch.quantile(t, p))
    return dict(n=len(g), rel_med=q(rel, .5), rel_p95=q(rel, .95), cos_med=q(cos, .5), cos_p05=q(cos, .05),
                cos_mean=float(cos.mean()), norm_err_med=q(dn, .5), norm_err_p95=q(dn, .95))


def accuracy(dec, codes, x):
    """float32 FD vs float32 autodiff; float64 autodiff and float64 FD used to separate truncation/round-off."""
    out = {}
    ctx = dec.prepare(codes).detach()
    g32, f32 = ad_grad(dec, ctx, x.clone(), False); g32, f32 = g32.detach(), f32.detach()
    dec64 = UnpackAttentionDecoder(dec.latent_dim, dec.width, dec.cross[0].attention.heads, dec.slots, len(dec.blocks),
                                   len(dec.cross)).to(x.device).double()
    dec64.load_state_dict({k: v.double() for k, v in dec.state_dict().items()}); dec64.requires_grad_(False)
    ctx64 = dec64.prepare(codes.detach().double())
    parts = []
    for i in range(0, x.shape[1], 1024):
        g, _ = ad_grad(dec64, ctx64, x[:, i:i + 1024].double().clone(), False); parts.append(g.detach())
    g64 = torch.cat(parts, 1)
    near = f32.abs() < .05
    gn = g32.norm(dim=-1)
    out['field'] = dict(f_abs_median=float(f32.abs().median()), f_abs_max=float(f32.abs().max()),
                        gradnorm_median=float(gn.median()), gradnorm_p05=float(torch.quantile(gn.flatten()[:1000000], .05)),
                        gradnorm_p95=float(torch.quantile(gn.flatten()[:1000000], .95)), near_fraction=float(near.float().mean()),
                        near_gradnorm_median=float(gn[near].median()) if int(near.sum()) else None)
    out['ad32_vs_ad64'] = vec_stats(g32, g64)
    rows = []
    with torch.no_grad():
        for method in ('central', 'tetra', 'forward'):
            for h in HS:
                g, _ = fd_grad(dec, ctx, x, method, h, chunk=8192)
                parts = [fd_grad(dec64, ctx64, x[:, i:i + 512].double(), method, h)[0] for i in range(0, x.shape[1], 512)]
                gd = torch.cat(parts, 1)
                rows.append(dict(method=method, h=h, vs_ad32=vec_stats(g, g32), vs_ad64=vec_stats(g, g64),
                                 near_vs_ad32=vec_stats(g, g32, near) if int(near.sum()) > 100 else None,
                                 fd64_vs_ad64=vec_stats(gd, g64),          # pure truncation error
                                 fd32_vs_fd64=vec_stats(g, gd)))           # pure round-off error
    out['rows'] = rows
    del dec64
    return out


def flat_grads(dec, codes):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in dec.parameters()]).double().clone(), codes.grad.flatten().double().clone()


def param_grad_agreement(dec, codes, x):
    step(dec, codes, x, 'autodiff', 0., False); ref, cref = flat_grads(dec, codes)
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    rows = []
    for method in ('central', 'tetra', 'forward'):
        for h in HS:
            eik, _ = step(dec, codes, x, method, h, False); g, c = flat_grads(dec, codes)
            rows.append(dict(method=method, h=h, cos_params=cos(g, ref), norm_ratio_params=float(g.norm() / ref.norm()),
                             rel_err_params=float((g - ref).norm() / ref.norm()), cos_codes=cos(c, cref), eik=eik))
    return dict(ref_norm=float(ref.norm()), rows=rows)


def stencil_sanity():
    a = torch.tensor([.3, -.7, .5]); lin = lambda p: p @ a
    out = {}
    for name, off in (('standard(1,1,1)', TETRA), ('as_written(1,1,-1)', TETRA_AS_WRITTEN)):
        out[name] = ((lin(off)[:, None] * off).sum(0) / 4).tolist()
    out['true'] = a.tolist()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True); p.add_argument('--run', type=Path); p.add_argument('--codes', type=Path)
    p.add_argument('--sections', default='time,acc,pgrad,amp'); p.add_argument('--iters', type=int, default=25)
    p.add_argument('--sizes', default='512:32,1024:32,4096:4,16384:1'); p.add_argument('--methods', default='autodiff,central,tetra,forward'); p.add_argument('--time-h', type=float, default=1e-2)
    a = p.parse_args(); sections = a.sections.split(','); dev = 'cuda'
    torch.manual_seed(0)
    res = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), box=[-BOX, BOX], hs=HS, time_h=a.time_h,
               tetra_sanity_linear=stencil_sanity(), cudnn_benchmark=torch.backends.cudnn.benchmark,
               tf32=torch.backends.cuda.matmul.allow_tf32, matmul_precision=torch.get_float32_matmul_precision())
    if a.run:
        from arm_planner.object_geometry.sdf_recipe import MeshSDF, RecipeConfig
        saved = torch.load(a.run / 'latest.pt', map_location='cpu', weights_only=False)
        model = MeshSDF(RecipeConfig(**saved['model_config'])); model.load_state_dict(saved['model'])
        dec = model.decoder.to(dev)
        codes = torch.load(a.codes)['codes'].to(dev)
        res.update(weights=f'trained {a.run}/latest.pt step {int(saved["step"])}', codes=f'encoder codes of 32 validation objects ({a.codes})')
        del model, saved
    else:
        dec = UnpackAttentionDecoder(32, 256, 8, 16, 2, 2, 256, 3).to(dev)
        codes = torch.randn(32, 32, device=dev) / 32 ** .5
        res.update(weights='random init (seed 0)', codes='randn(32,32)/sqrt(32)')
    dec.train()  # no dropout / batchnorm in this decoder; mode is irrelevant
    res['decoder_parameters'] = sum(p.numel() for p in dec.parameters())
    res['code_norm_median'] = float(codes.norm(dim=1).median())
    codes = codes.clone().requires_grad_(True)
    B = codes.shape[0]
    sample = lambda q: (torch.rand(B, q, 3, device=dev) * 2 - 1) * BOX
    save = lambda: a.out.write_text(json.dumps(res, indent=1))

    if 'acc' in sections:
        res['accuracy'] = accuracy(dec, codes.detach(), sample(4096)); save()
        print('accuracy done', flush=True)
    if 'pgrad' in sections:
        res['param_grad'] = {str(q): param_grad_agreement(dec, codes, sample(q)) for q in (1024,)}; save()
        print('pgrad done', flush=True)
    if 'time' in sections:
        res['timing'] = []
        for spec in a.sizes.split(','):
            q, MICRO[0] = [int(v) for v in spec.split(':')]
            x = sample(q)
            for projection in (False, True):
                for method in a.methods.split(','):
                    r = time_step(dec, codes, x, method, a.time_h, projection, a.iters)
                    r.update(queries=q, microbatch_codes=MICRO[0], loss='eik+proj' if projection else 'eik', method=method)
                    res['timing'].append(r); print(r, flush=True); save()
    if 'amp' in sections:
        res['amp'] = []
        x = sample(4096); MICRO[0] = 4
        for name, dt in (('bf16', torch.bfloat16), ('fp16', torch.float16)):
            for projection in (False, True):
                for method in ('autodiff', 'central', 'tetra', 'forward'):
                    try:
                        r = time_step(dec, codes, x, method, a.time_h, projection, a.iters, amp=dt)
                    except Exception as e:  # report, do not hide
                        r = dict(error=repr(e)[:300])
                    r.update(queries=4096, loss='eik+proj' if projection else 'eik', method=method, amp=name)
                    res['amp'].append(r); print(r, flush=True); save()
            # accuracy of gradients computed under autocast vs fp32 autodiff
            ctx = dec.prepare(codes.detach()).detach(); xs = sample(1024)
            g32, _ = ad_grad(dec, ctx, xs.clone(), False)
            try:
                with torch.autocast('cuda', dtype=dt):
                    ga, _ = ad_grad(dec, dec.prepare(codes.detach()), xs.clone(), False)
                r = dict(amp=name, kind='accuracy', method='autodiff', **vec_stats(ga.float(), g32))
                res['amp'].append(r); print(r, flush=True)
                for method in ('central', 'tetra', 'forward'):
                    for h in HS:
                        with torch.no_grad(), torch.autocast('cuda', dtype=dt):
                            g, _ = fd_grad(dec, dec.prepare(codes.detach()), xs, method, h)
                        r = dict(amp=name, kind='accuracy', method=method, h=h, **vec_stats(g.float(), g32))
                        res['amp'].append(r); print(r, flush=True)
            except Exception as e:
                res['amp'].append(dict(amp=name, kind='accuracy', error=repr(e)[:300]))
            save()
    save(); print('wrote', a.out)


if __name__ == '__main__':
    main()
