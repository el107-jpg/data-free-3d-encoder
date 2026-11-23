import json

from arm_planner.data_free.async_diagnostics import AsyncDiagnostics


def test_async_checkpoint_steps_and_restart(tmp_path):
    worker = tmp_path / "compute.py"
    worker.write_text("import json, os, sys, time\nfrom pathlib import Path\n"
                      "time.sleep(.1)\ns, r = map(Path,sys.argv[1:])\n"
                      "assert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
                      "r.write_text(json.dumps({'step': int(s.read_text())}))\n")
    records = []

    class Runner:
        step_count = 10
        def save_checkpoint(self, name):
            from pathlib import Path
            Path(name).write_text(str(self.step_count))

        def log_diagnostics(self, record):
            records.append((record["step"], self.step_count))
            with (tmp_path / "diagnostics.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")

    queue = AsyncDiagnostics(tmp_path, worker)
    runner = Runner()
    queue.schedule(runner)
    runner.step_count = 20
    queue.schedule(runner)
    assert len(records) < 2
    runner.step_count = 30
    queue.drain(runner)
    assert [r[0] for r in records] == [10, 20]
    assert records[-1][1] == 30
    assert json.loads((tmp_path / "diagnostic_result.json").read_text())["step"] == 20
    # A checkpoint persisted before a shutdown must be recovered on restart.
    (queue.queue / "00000030.pt").write_text("30")
    recovered = AsyncDiagnostics(tmp_path, worker)
    runner.step_count = 20
    recovered.schedule(runner)
    recovered.drain(runner)
    assert [r[0] for r in records] == [10, 20, 30]
    assert not list(queue.queue.glob("*.pt"))
