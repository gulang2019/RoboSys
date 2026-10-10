from __future__ import annotations

import ctypes
from contextlib import ExitStack
from itertools import combinations_with_replacement, permutations
from math import prod
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from ..policies import create_policy, PolicyConfig
from .environment import _cuda_call

if TYPE_CHECKING:
    from flash_rt.core.cuda_graph import CUDAGraph

stages = ('embed', 'encode', 'decode')
schedules = [
    (('embed',), ('encode',), ('decode',)),
    (('encode',), ('embed', 'decode')),
    (('encode', 'decode'), ('embed',)),
    (('embed', 'encode', 'decode'),),
]

def _get_device_name():
    return torch.cuda.get_device_name(0)


def _get_partitions(total: int, n_partition: int, divident=16) -> list[tuple[int, ...]]:
    """Positive, nondecreasing SM partitions that use all SMs."""
    if total <= 0 or n_partition <= 0 or divident <= 0 or total % divident:
        raise ValueError('total must be positive and divisible by divident')
    units = total // divident
    return [tuple(x * divident for x in partition)
            for partition in combinations_with_replacement(range(1, units + 1), n_partition)
            if sum(partition) == units]


def _get_streams(requested_sms, cleanup: ExitStack):
    """Allocate disjoint SM resources from one split; cleanup owns the handles."""
    from cuda.bindings import driver

    torch.cuda.init()
    device = _cuda_call(driver.cuDeviceGet, 0)
    resource = _cuda_call(driver.cuDeviceGetDevResource, device,
                         driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
    # The driver may round SM requests, so check the actual allocations.
    from math import gcd
    unit = gcd(*requested_sms)
    groups, count, _ = _cuda_call(driver.cuDevSmResourceSplitByCount,
                                  sum(requested_sms) // unit, resource, 0, unit)
    if count != sum(requested_sms) // unit or any(g.sm.smCount != unit for g in groups):
        raise RuntimeError(f'Unsupported SM partition: {requested_sms}')
    streams, offset = [], 0
    for sms in requested_sms:
        parts = groups[offset:offset + sms // unit]
        offset += sms // unit
        descriptor = _cuda_call(driver.cuDevResourceGenerateDesc, parts, len(parts))
        green = _cuda_call(driver.cuGreenCtxCreate, descriptor, device,
                          driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM)
        cleanup.callback(_cuda_call, driver.cuGreenCtxDestroy, green)
        raw = _cuda_call(driver.cuGreenCtxStreamCreate, green,
                        driver.CUstream_flags.CU_STREAM_NON_BLOCKING, 0)
        cleanup.callback(_cuda_call, driver.cuStreamDestroy, raw)
        stream = torch.cuda.ExternalStream(int(raw), device=0)
        stream.num_sms = sms
        cleanup.callback(stream.synchronize)
        streams.append(stream)
    return tuple(streams)


def _get_partition_streams(total: int, n_partitions, divident=16, *, cleanup: ExitStack):
    return [_get_streams(partition, cleanup)
            for partition in _get_partitions(total, n_partitions, divident)]


def _get_name(streams, schedule):
    labels = {'embed': 'EMB', 'encode': 'ENC', 'decode': 'DEC'}
    return ''.join('[' + ','.join(f'{labels[s]}{streams[s].num_sms}' for s in unit) + ']'
                   for unit in schedule)


@torch.inference_mode()
def _time(graphs: dict[str, CUDAGraph], streams: dict[str, torch.cuda.Stream],
          repeat=10, inputs=None, *, warmup=3, annotate=False) -> dict[str, float]:
    """Median seconds from a common GPU start to each stage's completion.

    Warm up before timing. Restore mutable inputs, then launch all
    graphs without a host synchronization between stages. Join every iteration.
    """
    if repeat < 1 or warmup < 0 or not graphs:
        raise ValueError('expected graphs, a positive repeat count and nonnegative warmup')
    control = torch.cuda.Stream(device=streams[next(iter(graphs))].device)
    samples = {stage: [] for stage in graphs}
    for iteration in range(repeat + warmup):
        # Copies are outside the timed region; the gate waits for all of them.
        for stage in graphs:
            stream = streams[stage]
            with torch.cuda.stream(stream):
                for dst, src in (inputs or {}).get(stage, ()):
                    dst.copy_(src)
                ready = torch.cuda.Event()
                ready.record(stream)
            control.wait_event(ready)
        start = torch.cuda.Event(enable_timing=True)
        ends = {stage: torch.cuda.Event(enable_timing=True) for stage in graphs}
        with torch.cuda.stream(control):
            # Allow the CPU to enqueue every replay before releasing the gate.
            torch.cuda._sleep(2_000_000)
            start.record(control)
        for stage, graph in graphs.items():
            stream = streams[stage]
            stream.wait_event(start)
            if annotate:
                with torch.cuda.nvtx.range(f'{stage}: {stream.num_sms} SMs; graph replay'):
                    graph.replay(ctypes.c_void_p(stream.cuda_stream))
            else:
                graph.replay(ctypes.c_void_p(stream.cuda_stream))
            ends[stage].record(stream)
        for end in ends.values():
            control.wait_event(end)
        control.synchronize()
        if iteration >= warmup:
            for stage, end in ends.items():
                samples[stage].append(start.elapsed_time(end) / 1000)
    return {stage: median(values) for stage, values in samples.items()}


def roofline(batch_sizes):
    for bsz in batch_sizes:
        for sch in schedules:
            t = 0
            delay = 0
            for unit in sch:
                flops, memacc = 0, 0
                for s in unit:
                    flops += perf_model.get_flops(s, bsz)
                    memacc += perf_model.get_memacc(s, bsz)
                _roofline_t = max(flops / hardware.flops, memacc / hardware.membw)
                delay += _roofline_t * len(unit)
                t += _roofline_t
            throughput = 1 / t * bsz
            ret.append((bsz, sch, delay, throughput))
    df = pd.DataFrame(ret)
    df.to_csv('profile/roofline.csv')

def _nsys_profile(graphs, streams, *, repeat=5, warmup=3, label='multi-stream'):
    """Capture warmed graph replays with nsys --capture-range=cudaProfilerApi.

    All stages run concurrently behind the same timing release gate.
    NVTX replay ranges mark CPU submissions; GPU kernels appear on their streams.
    """
    if repeat < 1 or warmup < 0:
        raise ValueError('repeat must be positive and warmup nonnegative')
    for _ in range(warmup):
        _time(graphs, streams, repeat=1, warmup=0)
    torch.cuda.synchronize()
    samples = []
    torch.cuda.profiler.start()
    try:
        with torch.cuda.nvtx.range(label):
            for iteration in range(repeat):
                with torch.cuda.nvtx.range(f'iteration {iteration}'):
                    samples.append(_time(graphs, streams, repeat=1, warmup=0, annotate=True))
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
    print(f'{label}: stage timings (ms)\n{pd.DataFrame(samples).mul(1000).describe().to_string()}', flush=True)
    return pd.DataFrame(samples).mean()

@torch.inference_mode()
def profile_single(batch_size, schedule: list[tuple[str, int]], *, repeat=5, warmup=3):
    """Trace a selected SM allocation, after graph compilation and warmup.

    For example, [('encode', 112), ('decode', 16)] runs both stages concurrently.
    [('encode', 112)] traces encode alone on a 112-SM green context.
    """
    selected_stages, sms = zip(*schedule)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    total_sms = torch.cuda.get_device_properties(device).multi_processor_count

    policy_config = PolicyConfig(
        model_name='pi05_libero', backend='flash_rt', precision='bf16',
        model_dir='checkpoints/pi05_libero_pytorch',
        batch_sizes={s: [batch_size] for s in stages},
        use_cuda_graph=True,
    )
    with ExitStack() as cleanup:
        common = torch.cuda.Stream(device)
        common.num_sms = total_sms
        streams = {stage: common for stage in stages}
        allocated = _get_streams(sms, cleanup)
        streams.update(zip(selected_stages, allocated))
        policy = create_policy(
            policy_config,
            device=device,
            streams={stage: [stream] for stage, stream in streams.items()},
        )
        cleanup.callback(policy.policy.close)
        cleanup.callback(torch.cuda.synchronize, device)
        graphs = {stage: policy._get_graph(stage, batch_size, streams[stage]) for stage in selected_stages}
        label = f'batch={batch_size} ' + ', '.join(f'{stage}={sms} SMs' for stage, sms in schedule)
        print(f'Capturing {label}: {warmup} warmup, {repeat} traced cycles', flush=True)
        _nsys_profile(graphs, streams, repeat=repeat, warmup=warmup, label=label)
@torch.inference_mode()
def profile_single_stage(batch_size, sms=None, *, repeat=5, warmup=3):
    device = torch.device('cuda:0')
    device_name = _get_device_name()
    torch.cuda.set_device(device)
    total_sms = torch.cuda.get_device_properties(device).multi_processor_count

    policy_config = PolicyConfig(
        model_name='pi05_libero', backend='flash_rt', precision='bf16',
        model_dir='checkpoints/pi05_libero_pytorch',
        batch_sizes={s: [batch_size] for s in stages},
        use_cuda_graph=True,
    )
    data = []
    with ExitStack() as cleanup:
        streams = [_get_streams((sm,), cleanup)[0]
                   for sm in (sms if sms is not None else range(16, total_sms + 1, 16))]
        policy = create_policy(
            policy_config,
            device=device,
            streams={stage: streams for stage in stages},
        )
        cleanup.callback(policy.policy.close)
        cleanup.callback(torch.cuda.synchronize, device)
        for stage in stages:
            for stream in streams:
                graphs = {stage: policy._get_graph(stage, batch_size, stream)}
                label = f'batch={batch_size} {stage}={stream.num_sms} SMs'
                timings = _nsys_profile(graphs, {stage: stream}, repeat=repeat, warmup=warmup, label=label)
                data.append((stage, device_name, batch_size, stream.num_sms / total_sms, timings[stage]))
    pd.DataFrame(data, columns=['stage', 'device', 'batch_size', 'sm_perc', 'time_s']).to_csv(
        'profile/profile_per_stage.csv', index=False)
    print('saved to profile/profile_per_stage.csv')
    
def profile(batch_sizes=[1,2,4,6,8]):
    batch_sizes = list(batch_sizes)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    device_name = _get_device_name()
    total_sms = torch.cuda.get_device_properties(device).multi_processor_count
    print('testing on', device_name, 'total sms', total_sms)
    
    policy_config = PolicyConfig(
        model_name='pi05_libero', backend='flash_rt', precision='bf16',
        model_dir='checkpoints/pi05_libero_pytorch',
        batch_sizes={s: batch_sizes for s in stages},
        use_cuda_graph=True,
    )
    tradeoffs = []
    with ExitStack() as cleanup:
        one_stage_stream = torch.cuda.Stream(device)
        one_stage_stream.num_sms = total_sms
        partitions = {1: [(one_stage_stream,)]}
        for count in (2, 3):
            partitions[count] = _get_partition_streams(total_sms, count, 16, cleanup=cleanup)
            print(f'number of {count} stream partitions: {len(partitions[count])}', flush=True)
        # Assign every distinct SM ordering to stages, without duplicate equal splits.
        options = {count: [streams for partition in groups
                           for streams in {tuple(s.num_sms for s in p): p
                                           for p in permutations(partition)}.values()]
                   for count, groups in partitions.items()}
        all_streams = [s for groups in partitions.values() for group in groups for s in group]
        print('total number of streams:', len(all_streams))

        total = len(batch_sizes) * sum(prod(len(options[len(unit)]) for unit in sch)
                                      for sch in schedules)
        progress = cleanup.enter_context(tqdm(total=total, desc='Loading policy', unit='config'))
        policy = create_policy(policy_config, device, streams={s: all_streams for s in stages})
        cleanup.callback(policy.policy.close)
        cleanup.callback(torch.cuda.synchronize, device)
        progress.set_description('Profiling')
        for bsz in batch_sizes:
            progress.set_postfix(batch=bsz, refresh=False)
            for sch in schedules:
                stream_options = [{}]
                for unit in sch:
                    stream_options = [x | dict(zip(unit, group))
                                      for x in stream_options for group in options[len(unit)]]
                # we calculate the roofline time here
                for streams in stream_options:
                    # Only this batch/stream combination is captured and retained.
                    policy.policy.compile({s: [streams[s]] for s in stages}, batch_size=bsz)
                    stage_times, delay = [], 0
                    for unit in sch:
                        graphs = {s: policy._get_graph(s, bsz, streams[s]) for s in unit}
                        times = _time(graphs, streams)
                        stage_times.append(max(times.values()))
                        delay += sum(times.values())
                    tradeoffs.append((device_name, bsz, _get_name(streams, sch), sch,
                                      delay, bsz / sum(stage_times)))
                    policy.policy.close()
                    progress.update(1)
            progress.write(f'finished batch size {bsz}: {len(tradeoffs)} results')

    output = Path('profile')
    output.mkdir(exist_ok=True)
    df = pd.DataFrame(tradeoffs, columns=['device', 'batch_size', 'name', 'schedule',
                                        'delay', 'throughput'])
    df.to_csv(output / 'multi_stream_tradeoffs.csv', index=False)


def roofline(batch_sizes):
    schedules = [
        (('embed',), ('encode',), ('decode',)),
        (('encode',), ('embed', 'decode')),
        (('encode', 'decode'), ('embed',)),
        (('embed', 'encode', 'decode'),),
    ]
    
    
    
def draw():
    from ast import literal_eval

    output = Path('profile')
    device_name = _get_device_name()
    df = pd.read_csv(
        output / 'multi_stream_tradeoffs.csv',
        converters={'schedule': literal_eval},
    )
    roofline_df = pd.read_csv(
        output / 'roofline.csv',
        converters={'schedule': literal_eval},
    )
    fig, ax = plt.subplots()
    for schedule, rows in df.groupby('schedule'):
        label = ' → '.join('+'.join(unit) for unit in schedule)
        ax.scatter(rows.delay * 1000, rows.throughput, label=label, s=12)
    for schedule, rows in roofline_df.groupby('schedule'):
        label = ' → '.join('+'.join(unit) for unit in schedule)
        ax.scatter(rows.delay * 1000, rows.throughput, marker = 'x', label=label + ' (roofline)', s=12)
        
    frontier = df.sort_values(['delay', 'throughput'], ascending=[True, False]).drop_duplicates('delay')
    frontier = frontier[frontier.throughput > frontier.throughput.cummax().shift(fill_value=-float('inf'))]
    ax.plot(frontier.delay * 1000, frontier.throughput, color='black', label='Pareto frontier')
    ax.set(xlabel='Delay (ms)', ylabel='Throughput (Inference/s)',
           title=device_name)
    ax.set_xlim(left = 25, right = 250)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'multi_stream_tradeoffs.png', dpi=150)
    plt.close(fig)
    print(f'saved to {output}/multi_stream_tradeoffs.png')

if __name__ == '__main__':
    # profile()
    # draw()
    profile_single_stage(2, [8, 16, 24, 32, 48, 64, 80, 96, 112, 128])
    
