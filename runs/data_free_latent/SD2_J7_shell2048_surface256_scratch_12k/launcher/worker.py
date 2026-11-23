from pathlib import Path
import argparse,gc,json,os,sys,subprocess
import torch
from gpu_budget import GpuBudget
from arm_planner.data_free.async_diagnostics import AsyncDiagnostics
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.train import Trainer
ROOT=Path('/home/eric-luo/Downloads/pointcloud-motion-planning')
BASE=ROOT/'runs/data_free_latent'
ORIGINAL=next(BASE.glob('J4_*/config.json'))
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(exist_ok=True,parents=True)
    cfg=DataFreeConfig.from_dict(json.loads(ORIGINAL.read_text()))
    cfg.surface_match_weight=1.0
    cfg.surface_sdf_probes=2048
    cfg.surface_start_queries=256
    cfg.surface_growth_interval=250
    assert cfg.seed==0 and cfg.lr==.0006364 and cfg.steps==5000 and not cfg.init_main_from
    cache=out/'checkpoint_last.pt'
    diagnostics = AsyncDiagnostics(out, Path(__file__).with_name('cpu_diagnostics.py'))
    with GpuBudget(out/'gpu.jsonl'):
        step=int(torch.load(cache,mmap=True,map_location='cpu',weights_only=False)['step']) if cache.exists() else 0
        if args.smoke and step:raise ValueError('preflight already exists')
        for end,rate in ([(2,.0006364)] if args.smoke else [(5000,.0006364),(10000,.0006364),(12000,.00006364)]):
            if step>=end:continue
            if cache.exists():
                overrides=[f'steps={end}', f'lr={rate}']
                if step >= 500:
                    overrides.append('surface_fixed_direction=true')
                runner=Trainer.from_checkpoint(str(cache),device='cuda',output_dir=str(out),overrides=overrides,tensorboard=not args.smoke)
                if step == 1300:
                    print('Resumed from checkpoint 1300 with parallel CPU surface matching and scoring during the first regularization pass', flush=True)
                if step == 500:
                    (out/'config_step500.json').write_text(json.dumps(runner.cfg.to_dict(),indent=2))
                    print('Switched to fixed Newton directions from checkpoint 500; same pair scores',flush=True)
            else:
                assert step==0
                runner=Trainer(cfg,device='cuda',output_dir=str(out),tensorboard=not args.smoke)
                (out/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2))
            print(f'phase {step} -> {end}; lr {rate}; scratch origin seed=0; cuda allocator fraction=0.70',flush=True)
            # Preserve comparison checkpoints without duplicating their write in device memory.
            save=runner.save_checkpoint
            def keep(path=None):
                name=save(path)
                if runner.step_count in (250,500,1000,2500,4000,5000,10000,10500,11000,12000):
                    dest=out/f'checkpoint_{runner.step_count}.pt'
                    if not dest.exists():os.link(name,dest)
                return name
            runner.save_checkpoint=keep
            log_step=runner.log_step
            def log_and_verify(record,scalars):
                log_step(record,scalars)
                diagnostics.poll(runner)
                if runner.step_count <= step + 5:
                    runner._metrics_file.flush()
                    if record['skipped'] or not all(torch.isfinite(torch.tensor(record[key])) for key in ('loss','grad_norm','surface/match_distance','surface/point_gradient_norm','surface/field_distance')):
                        raise RuntimeError('failed numerical check during startup')
                    if (record['surface/sdf_probes'] != 2048 or record['surface/shell_probes'] != 1024
                            or record['surface/point_gradient_norm'] <= 0 or record['surface/pair_coverage'] <= 0):
                        raise RuntimeError('surface matching or shared shell sampling failed startup check')
                    if step >= 1300:
                        assert 0 <= record['time_ms/diversity_wait'] and 0 < record['time_ms/diversity_scores']
                    if runner.cfg.surface_fixed_direction:
                        assert record['surface/fixed_direction'] == 1 and record['surface/match_source_gradient_norm'] > 0
                    print(f"update {runner.step_count} field {record['surface/field_distance']:.5f}, "
                          f"surface {record['surface/match_distance']:.5f}, queries {record['surface/query_mean']:.1f}, "
                          f"pair coverage {record['surface/pair_coverage']:.3f}",flush=True)
                if runner.step_count in (5, 505, 1305) and not args.smoke:
                    runner.save_checkpoint()
                    print('verified first five updates and saved a resume checkpoint',flush=True)
            runner.log_step=log_and_verify
            def schedule_diagnostic(verbose=True):
                result = diagnostics.schedule(runner)
                if runner.step_count == 12000:
                    diagnostics.drain(runner)
                return result
            runner._diagnose = schedule_diagnostic
            runner._open_logs()
            # Validate concurrent publication at rollout, then follow J7's original cadence.
            if runner.step_count == 300 and not args.smoke:
                schedule_diagnostic()
            else:
                diagnostics.poll(runner)
            if args.smoke: runner.cfg.diag_every = 0
            runner.run(steps=end-step)
            step=runner.step_count
            del keep,save,log_and_verify,log_step,runner
            gc.collect();torch.cuda.empty_cache()
    print('finished',step,flush=True)
if __name__=='__main__':main()
