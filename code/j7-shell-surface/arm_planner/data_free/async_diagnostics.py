"""Run the existing checkpoint diagnostic on a single CPU worker alongside training."""
import json
import os
from pathlib import Path
import subprocess
import sys


class AsyncDiagnostics:
    def __init__(self, output, worker):
        self.output = Path(output)
        self.worker = Path(worker)
        self.queue = self.output / "diagnostic_queue"
        self.queue.mkdir(exist_ok=True)
        log = self.output / "diagnostics.jsonl"
        self.logged = {json.loads(line)["step"] for line in log.open()} if log.exists() else set()
        self.process = None
        self.snapshot = None
        self.failures = {}

    def schedule(self, runner):
        step = runner.step_count
        snapshot = self.queue / f"{step:08d}.pt"
        if step not in self.logged and not snapshot.exists():
            runner.save_checkpoint(str(snapshot))
            print(f"Queued CPU diagnostic at step {step}", flush=True)
        self.poll(runner)
        return step

    def poll(self, runner, wait=False):
        """Only the trainer publishes metrics, preserving original step labels and order."""
        if self.process is not None:
            code = self.process.wait() if wait else self.process.poll()
            if code is None:
                return
            previous = self.snapshot
            self.process = self.snapshot = None
            if code:
                self.failures[previous] = self.failures.get(previous, 0) + 1
                print(f"CPU diagnostic for {previous.stem} exited {code}; snapshot retained", flush=True)
        for snapshot in sorted(self.queue.glob("*.pt")):
            step = int(snapshot.stem)
            result = snapshot.with_suffix(".json")
            if step in self.logged:
                snapshot.unlink()
                result.unlink(missing_ok=True)
                continue
            if result.is_file():
                record = json.loads(result.read_text())
                if record["step"] != step:
                    raise ValueError(f"Wrong diagnostic step in {result}")
                runner.log_diagnostics(record)
                self.logged.add(step)
                os.replace(result, self.output / "diagnostic_result.json")
                snapshot.unlink()
                print(f"Published CPU diagnostic for step {step} at training step {runner.step_count}", flush=True)
                continue
            if self.failures.get(snapshot, 0) >= 2:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="6", MKL_NUM_THREADS="6")
            self.process = subprocess.Popen([sys.executable, str(self.worker), str(snapshot), str(result)],
                                            env=env)
            self.snapshot = snapshot
            return

    def drain(self, runner):
        self.poll(runner)
        while self.process is not None:
            self.poll(runner, wait=True)
        if any(self.queue.glob("*.pt")):
            raise RuntimeError(f"CPU diagnostics need recovery from snapshots in {self.queue}")
