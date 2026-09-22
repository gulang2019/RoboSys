from pathlib import Path
from itertools import product
from dataclasses import asdict, fields
import ast
import argparse
import csv
import pprint

from robort.profile.runner import Runner
from robort.profile.schemas import (
    RunnerConfig,
    HardwareConfig)
from robort.policies import PolicyConfig

import logging

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES = {
    'embed': [1,2,4,8],
    'encode': [1,2],
    'decode': [1,2,4,8,16]
}

HW_CHOICES = [
    ('power_perc', [1.0]),
    ('sm_perc', [1.0])
]

POLICY_CHOICES = [
    ('model_name', ['pi05_libero']),
    ('backend', ['openpi']),
    ('num_views', [1,2,3]),
    ('image_resolution', [224,368,512]),
    ('precision', ['bf16']),
    ('batch_sizes', [DEFAULT_BATCH_SIZES]),
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
    
    print("=" * 20)
    print("hardware choices")
    pprint.pprint(_hw_choices)
    print("policy choices")
    pprint.pprint(policy_choices)
    print("=" * 20)

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
                    if 'use_torch_compile' in keys:
                        row.setdefault('use_torch_compile', 'True')
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

        for policy_config in policy_configs: 
            runner.init_backend(policy_config, hw_configs)
            for hw_config in hw_configs:
                profile = runner.profile_policy(
                    hardware_config=hw_config, policy_config=policy_config)
                policy_rows = []
                config = {**asdict(profile.hardware_config), **asdict(profile.policy_config)}
                config.pop("batch_sizes")
                for stage, measurements in profile.stages.items():
                    for batch_size, (lat_mean, lat_std) in measurements.lat.items():
                        energy_mean, energy_std = measurements.energy[batch_size]
                        memory_mean, memory_std = measurements.mem_fp_activation_gb[batch_size]
                        policy_rows.append({**config, 'batch_size': batch_size, 'stage': stage,
                                            'num_params': measurements.num_params, 'flops': measurements.flops,
                                            'mem_fp_weight_gb': measurements.mem_fp_weight_gb,
                                            'lat_mean': lat_mean, 'lat_std': lat_std,
                                            'energy_mean': energy_mean, 'energy_std': energy_std,
                                            'mem_fp_activation_gb_mean': memory_mean,
                                            'mem_fp_activation_gb_std': memory_std})
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
        if name == 'batch_sizes': continue 
        parser.add_argument(
            '--' + name.replace('_', '-'), type=type(values[0]), nargs='+',
            default= values, help=f'Sweep values for {name} (default: {values}).')
    parser.add_argument('--model-dir', nargs='+')
    parser.add_argument('--num-steps', type=int, nargs='+')
    parser.add_argument('--chunk-size', type=int, nargs='+')
    parser.add_argument('--use-cuda-graph', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--use-torch-compile', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable torch.compile before CUDA graph capture (default: enabled)')
    for stage in DEFAULT_BATCH_SIZES:
        parser.add_argument(f'--{stage}-batch-sizes', type=int, nargs='+', default= DEFAULT_BATCH_SIZES[stage],
                            help='Override --batch-sizes for this stage’s measurements')
    args = parser.parse_args()
    if args.num_warmup < 0:
        parser.error('--num-warmup must be >= 0')
    if args.num_iter < 1:
        parser.error('--num-iter must be >= 1')
    for name, _ in HW_CHOICES:
        if any(not 0 < value <= 1 for value in getattr(args, name)):
            parser.error(f'--{name.replace("_", "-")} values must be in (0, 1]')
    policy_choices = {name: getattr(args, name) for name, _ in POLICY_CHOICES if hasattr(args, name)}
    batch_sizes = {stage: getattr(args, f'{stage}_batch_sizes') for stage in DEFAULT_BATCH_SIZES}
    policy_choices['batch_sizes'] = [batch_sizes]
    
    for name in ('model_dir', 'num_steps', 'chunk_size'):
        if getattr(args, name) is not None:
            policy_choices[name] = getattr(args, name)
    if args.use_cuda_graph is not None:
        policy_choices['use_cuda_graph'] = [args.use_cuda_graph]
    if args.use_torch_compile is not None:
        policy_choices['use_torch_compile'] = [args.use_torch_compile]
    logging.basicConfig(level=logging.INFO)

    main(
        RunnerConfig(num_warmup=args.num_warmup, num_iter=args.num_iter,
                     output_dir=args.output_dir,
                     ),
        {name: getattr(args, name) for name, _ in HW_CHOICES},
        policy_choices,
    )
