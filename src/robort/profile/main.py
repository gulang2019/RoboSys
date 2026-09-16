from pathlib import Path
from itertools import product
from dataclasses import asdict, fields
import ast
import argparse
import csv

from robort.profile.runner import Runner
from robort.profile.schemas import (
    RunnerConfig,
    HardwareConfig)
from robort.policies import PolicyConfig

import logging

logger = logging.getLogger(__name__)

HW_CHOICES = [
    ('power_perc', [0.2,0.4,0.6,0.8,1.0]),
    ('sm_perc', [0.2,0.4,0.6,0.8,1.0])
]

POLICY_CHOICES = [
    ('model_name', ['pi05_libero']),
    ('backend', ['openpi', 'flash_rt']),
    ('num_views', [1,2,3]),
    ('image_resolution', [224,368,512]),
    ('precision', ['bf16', 'fp16', 'fp8']),
    ('batch_sizes', [[1,2,3,4,5,6,7,8]]),
    ('prompt_len', [10])
]

def main(runner_config: RunnerConfig,
          hw_choices: dict[str, list],
          policy_choices: dict[str, list]):
    '''
    Perform a sweep over different hardware and policy configurations.
    '''
    _hw_choices = dict(HW_CHOICES)
    _hw_choices |= hw_choices
    _policy_choices = dict(POLICY_CHOICES)
    _policy_choices |= policy_choices

    hw_fields, hw_choicess = zip(*_hw_choices.items())
    hw_configs = [HardwareConfig(**dict(zip(hw_fields, choices)))
                  for choices in product(*hw_choicess)]

    policy_fields, policy_choicess = zip(*_policy_choices.items())
    policy_configs = [PolicyConfig(**dict(zip(policy_fields, choices)))
                      for choices in product(*policy_choicess)]

    logger.info(f"Sweep {len(hw_configs)} hardware config, {len(policy_configs)} Policy Configs")
    runner = Runner(runner_config)
    output_dir = Path(runner_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware_keys = [field.name for field in fields(HardwareConfig)]
    policy_keys = ["batch_size" if field.name == "batch_sizes" else field.name
                   for field in fields(PolicyConfig)]

    def store_profiles(path, rows, keys):
        if not rows:
            return
        merged = {}
        columns = []
        if path.exists():
            with path.open(newline='') as source:
                reader = csv.DictReader(source)
                columns = ["batch_size" if key == "batch_sizes" else key
                           for key in (reader.fieldnames or [])]
                for row in reader:
                    if "batch_sizes" in row:
                        sizes = ast.literal_eval(row.pop("batch_sizes"))
                        if not isinstance(sizes, list) or len(sizes) != 1:
                            raise ValueError(f"Cannot migrate ambiguous batch_sizes in {path}: {sizes!r}")
                        row["batch_size"] = str(sizes[0])
                    merged[tuple(row[key] for key in keys)] = row
        for row in rows:
            row = {key: '' if value is None else str(value)
                   for key, value in row.items()}
            merged[tuple(row[key] for key in keys)] = row
            columns.extend(key for key in row if key not in columns)
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=columns)
            writer.writeheader()
            writer.writerows(merged.values())
        temporary.replace(path)

    try:
        for hw_config in hw_configs:
            try:
                profile = runner.profile_hardware(hw_config)
            except Exception as e:
                logger.warning('[Hardware profile] Skip %s: %s', hw_config, e)
                continue
            row = asdict(profile)
            config = row.pop('hardware_config')
            store_profiles(output_dir / 'hardware_profile.csv', [{**config, **row}], hardware_keys)

        for hw_config, policy_config in product(hw_configs, policy_configs):
            try:
                profiles = runner.profile_policy(
                    hardware_config=hw_config, policy_config=policy_config)
            except Exception as e:
                logger.warning('[Policy profile] Skip %s, %s: %s', hw_config, policy_config, e)
                continue
            policy_rows = []
            for batch_size, profile in profiles.items():
                config = {**asdict(profile.hardware_config), **asdict(profile.policy_config)}
                config.pop("batch_sizes")
                config["batch_size"] = batch_size
                for stage, measurements in profile.stages.items():
                    policy_rows.append({**config, 'stage': stage, **asdict(measurements)})
            store_profiles(output_dir / 'policy_profile.csv', policy_rows,
                           hardware_keys + policy_keys + ['stage'])
    finally:
        runner.close()
    logger.info('results saved under %s', output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sweep hardware and policy profiling configurations.')
    parser.add_argument('--num-warmup', type=int, default=RunnerConfig.num_warmup)
    parser.add_argument('--num-iter', type=int, default=RunnerConfig.num_iter)
    parser.add_argument('--output-dir', default=RunnerConfig.output_dir)
    for name, values in HW_CHOICES + POLICY_CHOICES:
        parser.add_argument(
            '--' + name.replace('_', '-'), type=int if name == 'batch_sizes' else type(values[0]), nargs='+',
            default=values[0] if name == 'batch_sizes' else values, help=f'Sweep values for {name} (default: {values}).')
    parser.add_argument('--model-dir', nargs='+')
    parser.add_argument('--num-steps', type=int, nargs='+')
    parser.add_argument('--chunk-size', type=int, nargs='+')
    parser.add_argument('--use-cuda-graph', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.num_warmup < 0:
        parser.error('--num-warmup must be >= 0')
    if args.num_iter < 1:
        parser.error('--num-iter must be >= 1')
    for name, _ in HW_CHOICES:
        if any(not 0 < value <= 1 for value in getattr(args, name)):
            parser.error(f'--{name.replace("_", "-")} values must be in (0, 1]')
    policy_choices = {name: getattr(args, name) for name, _ in POLICY_CHOICES}
    policy_choices['batch_sizes'] = [args.batch_sizes]
    for name in ('model_dir', 'num_steps', 'chunk_size'):
        if getattr(args, name) is not None:
            policy_choices[name] = getattr(args, name)
    if args.use_cuda_graph is not None:
        policy_choices['use_cuda_graph'] = [args.use_cuda_graph]
    logging.basicConfig(level=logging.INFO)
    main(
        RunnerConfig(num_warmup=args.num_warmup, num_iter=args.num_iter,
                     output_dir=args.output_dir),
        {name: getattr(args, name) for name, _ in HW_CHOICES},
        policy_choices,
    )
