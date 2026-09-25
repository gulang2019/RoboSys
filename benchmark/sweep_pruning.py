"""Sweep diverse LIBERO tasks with one local OpenPI subprocess per GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
from fractions import Fraction
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ('libero_spatial', 0, 'bowl_between_objects', 'pick up the black bowl between the plate and the ramekin and place it on the plate'),
    ('libero_object', 7, 'milk_into_basket', 'pick up the milk and place it in the basket'),
    ('libero_goal', 5, 'push_plate', 'push the plate to the front of the stove'),
    ('libero_10', 3, 'bowl_into_drawer_and_close', 'put the black bowl in the bottom drawer of the cabinet and close it'),
)
RATES = ('1/6', '1/3', '1/2', '2/3')
STEPS = (2, 4, 7, 10)
ARTIFACTS = ('combined.mp4', 'scores.cdf.png', 'attention.cdf.png')



class ModelWorker:
    """Lazily start one interpreter/model per GPU and restart after failures."""

    def __init__(self, python, env):
        self.python, self.env = python, env
        self.process = None

    def run(self, command, log):
        if self.process is None:
            self.process = subprocess.Popen(
                [self.python, '-u', '-m', 'benchmark.sweep_worker'],
                cwd=ROOT, env=self.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True,
            )
        try:
            self.process.stdin.write(json.dumps(dict(argv=command[4:], log=str(log))) + '\n')
            self.process.stdin.flush()
            response = self.process.stdout.readline()
            code = json.loads(response)['returncode'] if response else 1
        except (OSError, ValueError):
            self.close()
            raise
        if code:
            self.close()
        return code

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
            self.process.stdout.close()
            self.process = None


def completed_result(attempt, episodes):
    try:
        result = json.loads((attempt / 'results.json').read_text())
        if result['episodes'] != episodes or not 0 <= result['successes'] <= episodes:
            return None
        if not all((attempt / 'debug' / f'episode_{i:03d}' / name).is_file()
                   and (attempt / 'debug' / f'episode_{i:03d}' / name).stat().st_size > 0
                   for i in range(episodes) for name in ARTIFACTS):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/libero' / datetime.now().strftime('pruning_sweep_%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--replan-steps', type=int, default=5)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/pi05_libero_pytorch')
    parser.add_argument('--python', default=str(ROOT / '.venv-openpi/bin/python'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1 or not 1 <= args.replan_steps <= 10 or len(set(args.gpus)) != len(args.gpus):
        parser.error('episodes must be positive, replan-steps must be 1..10, and GPUs must be unique')
    output = args.output_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    manifest = dict(tasks=TASKS, keep_rates=RATES, decoding_steps=STEPS,
                    episodes=args.episodes, seed=args.seed, replan_steps=args.replan_steps,
                    checkpoint=str(checkpoint), python=args.python)
    manifest_text = json.dumps(manifest, indent=2) + '\n'
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            parser.error('output directory is not empty; use --resume or a new --output-dir')
        if not (output / 'manifest.json').is_file() or (output / 'manifest.json').read_text() != manifest_text:
            parser.error('resume settings differ from the saved manifest')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(manifest_text)
    rows = []
    for suite, task_id, name, description in TASKS:
        for rate in RATES:
            for steps in STEPS:
                run_dir = output / suite / f'task_{task_id:03d}_{name}' / f'keep_{rate.replace("/", "of")}' / f'decode_{steps:02d}'
                rows.append(dict(suite=suite, task_id=task_id, task=description,
                                 keep_fraction=rate, encode_keep_rate=float(Fraction(rate)),
                                 num_steps=steps, requested_episodes=args.episodes,
                                 seed=args.seed, replan_steps=args.replan_steps,
                                 gpu='', status='pending', episodes='', successes='', success_rate='',
                                 run_dir=str(run_dir), attempt_dir='', error=''))
    lock = Lock()

    def save_csv():
        temporary = output / 'success_rates.csv.tmp'
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / 'success_rates.csv')

    save_csv()
    print(f'{len(rows)} configurations, {len(rows) * args.episodes} episodes, GPUs {args.gpus}', flush=True)
    print(f'Results: {output}', flush=True)
    if args.dry_run:
        return 0

    jobs = Queue()
    for row in rows:
        jobs.put(row)

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MUJOCO_GL='egl', JAX_PLATFORMS='cpu',
                   PYTHONPATH=f'{ROOT}:{ROOT / "src"}:{ROOT / "3rdparty/AutoHorizon/third_party/libero"}',
                   LIBERO_CONFIG_PATH=os.environ.get('LIBERO_CONFIG_PATH', str(ROOT / 'data/libero_config')))
        model_worker = ModelWorker(args.python, env)
        try:
            while True:
                try:
                    row = jobs.get_nowait()
                except Empty:
                    return
                run_dir = Path(row['run_dir'])
                attempts = sorted(run_dir.glob('attempt_*'))
                previous = next(((a, r) for a in reversed(attempts)
                                 if (r := completed_result(a, args.episodes)) is not None), None)
                if previous:
                    attempt, result = previous
                    error = ''
                else:
                    attempt = run_dir / f'attempt_{len(attempts) + 1:03d}'
                    attempt.mkdir(parents=True)
                    client = dict(type='local', device='cuda:0', model_dir=str(checkpoint),
                                  num_steps=row['num_steps'], encode_keep_rate=row['encode_keep_rate'],
                                  debug_dir=str(attempt / 'debug'))
                    command = [args.python, '-u', '-m', 'benchmark.libero',
                               '--client-args', json.dumps(client),
                               '--args.task-suite-name', row['suite'], '--args.task-id', str(row['task_id']),
                               '--args.num-trials-per-task', str(args.episodes), '--args.seed', str(args.seed),
                               '--args.replan-steps', str(args.replan_steps), '--args.fail-on-error',
                               '--args.results-path', str(attempt / 'results.json'),
                               '--args.video-out-path', str(attempt / 'rollouts')]
                    (attempt / 'command.json').write_text(json.dumps(dict(gpu=gpu, argv=command), indent=2))
                    with lock:
                        row.update(gpu=gpu, status='running', attempt_dir=str(attempt))
                        save_csv()
                    try:
                        code = model_worker.run(command, attempt / 'eval.log')
                        result = completed_result(attempt, args.episodes) if code == 0 else None
                        error = '' if result is not None else f'exit={code}; see eval.log / check artifacts'
                    except OSError as exc:
                        result, error = None, str(exc)
                with lock:
                    row.update(status='complete' if result is not None else 'failed', attempt_dir=str(attempt), error=error)
                    if previous:
                        row['gpu'] = json.loads((attempt / 'command.json').read_text())['gpu']
                    if result is not None:
                        row.update(episodes=result['episodes'], successes=result['successes'],
                                   success_rate=result['successes'] / result['episodes'])
                    save_csv()
                    print(f'GPU {gpu}: {row["suite"]}/{row["task_id"]} keep={row["keep_fraction"]} '
                          f'steps={row["num_steps"]}: {row["status"]} {row["success_rate"]}', flush=True)
        finally:
            model_worker.close()

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    return int(any(row['status'] != 'complete' for row in rows))


if __name__ == '__main__':
    sys.exit(main())
