from pathlib import Path
import argparse,gc,json,os,sys,subprocess
import torch
from gpu_budget import GpuBudget
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
    cfg.slope_diversity_weight=.04
    assert cfg.seed==0 and cfg.lr==.0006364 and cfg.steps==5000 and not cfg.init_main_from
    cache=out/'checkpoint_last.pt'
    with GpuBudget(out/'gpu.jsonl'):
        step=int(torch.load(cache,mmap=True,map_location='cpu',weights_only=False)['step']) if cache.exists() else 0
        if args.smoke and step:raise ValueError('preflight already exists')
        for end,rate in ([(2,.0006364)] if args.smoke else [(5000,.0006364),(10000,.0006364),(12000,.00006364)]):
            if step>=end:continue
            if cache.exists():
                runner=Trainer.from_checkpoint(str(cache),device='cuda',output_dir=str(out),overrides=[f'steps={end}',f'lr={rate}'],tensorboard=not args.smoke)
            else:
                assert step==0
                runner=Trainer(cfg,device='cuda',output_dir=str(out),tensorboard=not args.smoke)
                (out/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2))
            print(f'phase {step} -> {end}; lr {rate}; scratch origin seed=0; cuda allocator fraction=0.70',flush=True)
            # Preserve comparison checkpoints without duplicating their write in device memory.
            save=runner.save_checkpoint
            def keep(path=None):
                name=save(path)
                if runner.step_count in (1000,2500,4000,5000,10000,10500,11000,12000):
                    dest=out/f'checkpoint_{runner.step_count}.pt'
                    if not dest.exists():os.link(name,dest)
                return name
            runner.save_checkpoint=keep
            log_step=runner.log_step
            def log_and_verify(record,scalars):
                log_step(record,scalars)
                if runner.step_count <= step + 5:
                    runner._metrics_file.flush()
                    if record['skipped'] or not all(torch.isfinite(torch.tensor(record[key])) for key in ('loss','grad_norm','div/slope_angle')):
                        raise RuntimeError('failed numerical check during startup')
                if runner.step_count == 5 and not args.smoke:
                    runner.save_checkpoint()
                    print('verified first five updates and saved the initial resume checkpoint',flush=True)
            runner.log_step=log_and_verify
            def cpu_diagnostic():
                # Diagnostics have a separate RNG and do not update model or optimizer state.
                snap = out/'diagnostic_state.pt'
                result = out/'diagnostic_result.json'
                runner.save_checkpoint(str(snap))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6')
                print(f'CPU diagnostic at step {runner.step_count}', flush=True)
                subprocess.run([sys.executable, str(Path(__file__).with_name('cpu_diagnostics.py')), str(snap), str(result)], env=env, check=True)
                rec = json.loads(result.read_text())
                assert rec['step'] == runner.step_count
                snap.unlink()
                return rec
            runner.run_diagnostics = cpu_diagnostic
            runner.run(steps=end-step)
            step=runner.step_count
            del keep,save,log_and_verify,log_step,runner
            gc.collect();torch.cuda.empty_cache()
    print('finished',step,flush=True)
if __name__=='__main__':main()
