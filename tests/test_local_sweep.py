import csv
import json
from pathlib import Path
import sys
from threading import Barrier, Lock
from types import SimpleNamespace

from benchmark import sweep_local


def test_two_gpu_sweep_and_resume(monkeypatch, tmp_path):
    output = tmp_path / 'sweep'
    argv = ['sweep', '--output-dir', str(output), '--episodes', '1']
    monkeypatch.setattr(sys, 'argv', argv)
    calls = []
    active = set()
    lock = Lock()
    barrier = Barrier(2)
    failed_once = False

    def run(self, command, log):
        env = self.env
        log.touch()
        nonlocal failed_once
        gpu = env['CUDA_VISIBLE_DEVICES']
        client = json.loads(command[command.index('--client-args') + 1])
        assert client['device'] == 'cuda:0'
        assert env['JAX_PLATFORMS'] == 'cpu'
        with lock:
            assert gpu not in active
            active.add(gpu)
            first = not any(call[0] == gpu for call in calls)
            calls.append((gpu, client))
            fail = not failed_once and gpu == '0'
            failed_once |= fail
        if first:
            barrier.wait(timeout=10)
        if not fail:
            path = Path(command[command.index('--args.results-path') + 1])
            path.write_text(json.dumps(dict(episodes=1, successes=1)))
            episode = Path(client['debug_dir']) / 'episode_000'
            episode.mkdir(parents=True)
            for artifact in sweep_local.ARTIFACTS:
                (episode / artifact).write_bytes(b'fake artifact')
        with lock:
            active.remove(gpu)
        return 1 if fail else 0

    monkeypatch.setattr(sweep_local.ModelWorker, 'run', run)
    assert sweep_local.main() == 1
    assert len(calls) == 64
    assert {gpu for gpu, _ in calls} == {'0', '1'}
    with (output / 'success_rates.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 64
    assert sum(row['status'] == 'complete' for row in rows) == 63
    failed = next(row for row in rows if row['status'] == 'failed')
    old_attempt = Path(failed['attempt_dir'])
    assert (old_attempt / 'eval.log').exists()
    monkeypatch.setattr(sys, 'argv', argv + ['--resume'])
    assert sweep_local.main() == 0
    assert len(calls) == 65
    assert old_attempt.exists()
    with (output / 'success_rates.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(row['status'] == 'complete' and row['success_rate'] == '1.0' for row in rows)
    assert sum('attempt_002' in row['attempt_dir'] for row in rows) == 1


def test_dry_run_default_coverage(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['sweep', '--output-dir', str(tmp_path), '--dry-run'])
    assert sweep_local.main() == 0
    with (tmp_path / 'success_rates.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 64
    assert sum(int(row['requested_episodes']) for row in rows) == 640
    assert {row['num_steps'] for row in rows} == {'2', '4', '7', '10'}
    assert {row['keep_fraction'] for row in rows} == {'1/6', '1/3', '1/2', '2/3'}



def test_worker_reuses_process_and_restarts_after_failure(monkeypatch, tmp_path):
    from io import StringIO
    from unittest.mock import Mock

    processes = []
    def popen(*args, **kwargs):
        process = Mock(stdin=StringIO(), stdout=StringIO(
            '{"returncode": 0}\n{"returncode": 1}\n'))
        processes.append(process)
        return process

    monkeypatch.setattr(sweep_local.subprocess, 'Popen', popen)
    worker = sweep_local.ModelWorker('python', {})
    command = ['python', '-u', '-m', 'benchmark.libero', '--client-args', '{}']
    assert worker.run(command, tmp_path / 'first.log') == 0
    assert worker.run(command, tmp_path / 'second.log') == 1
    assert len(processes) == 1
    processes[0].wait.assert_called_once()
    assert worker.run(command, tmp_path / 'third.log') == 0
    assert len(processes) == 2
    worker.close()
    processes[1].wait.assert_called_once()


def test_persistent_worker_protocol_with_two_configurations(tmp_path):
    """Exercise real pipes, tyro parsing, and log redirection without CUDA."""
    import subprocess

    script = """
import dataclasses, sys, types
client_module = types.ModuleType('benchmark.client')
libero_module = types.ModuleType('benchmark.libero')
@dataclasses.dataclass
class Args:
    seed: int = 7
class Client:
    def __init__(self, **options):
        print('MODEL LOADED', flush=True)
    def configure_run(self, **options):
        print('RECONFIGURED', options['num_steps'], flush=True)
    def close(self):
        pass
def evaluate(args, client_args, client=None):
    assert client is not None
    print('EVALUATED', args.seed, flush=True)
import json
client_module.LocalClientPolicy = Client
client_module.parse_client_args = lambda value: ('local', json.loads(value))
libero_module.Args = Args
libero_module._eval_libero = evaluate
sys.modules['benchmark.client'] = client_module
sys.modules['benchmark.libero'] = libero_module
from benchmark.sweep_worker import main
main()
"""
    jobs = [
        dict(argv=['--client-args', json.dumps(dict(
            num_steps=steps, encode_keep_rate=0.5, debug_dir=str(tmp_path))),
            '--args.seed', '7'], log=str(tmp_path / f'{steps}.log'))
        for steps in (2, 4)
    ]
    result = subprocess.run([sys.executable, '-c', script],
                            input=''.join(json.dumps(job) + '\n' for job in jobs),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        {'returncode': 0}, {'returncode': 0}]
    assert 'MODEL LOADED' in (tmp_path / '2.log').read_text()
    assert 'MODEL LOADED' not in (tmp_path / '4.log').read_text()
    assert 'RECONFIGURED 4' in (tmp_path / '4.log').read_text()
    assert all('EVALUATED 7' in (tmp_path / f'{steps}.log').read_text() for steps in (2, 4))
