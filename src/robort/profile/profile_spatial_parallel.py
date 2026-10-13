import itertools
from contextlib import ExitStack
import torch 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from dataclasses import dataclass

import triton 
from ..kernels.gemm import matmul_kernel

DEVICE = torch.device('cuda:0')

@dataclass 
class Kernel:
    @property 
    def f(self): # Gflops
        ...
    @property
    def m(self): # G byte 
        ...
    
    @staticmethod
    def _nelem(dtype: str):
        return {
            'bfloat16': 2,
            'float16': 2 
        }[dtype]
    
    def run(self):
        ...
    
    @property 
    def cycle(self):
        ...
        
    @property
    def impl(self):
        ...

    def get_autotune_configs(self):
        raise NotImplementedError

    def set_config(self, config):
        raise NotImplementedError

class Device: 
    def __init__(self, name: str):
        csv = 'profile/hardware.csv'
        df = pd.read_csv(csv, index_col = 0)
        self.F = df.loc[name, "tflops"] * 1e3
        self.B = df.loc[name, "membw"]
    
    def tc(self, kernel: Kernel):
        return kernel.f / self.F
    
    def tm(self, kernel: Kernel):
        return kernel.m / self.B
    
    def t(self, kernel: Kernel):
        return max(self.tc(kernel), self.tm(kernel))

def get_roofline(
    a: Kernel, 
    b: Kernel,
    d: Device,
    _multiple: int = 5):
    
    tas = []
    tbs = []
    for ta in np.linspace(d.t(a) + 1e-6, d.t(a) * _multiple, 50):
        tb = max(d.tc(b)* (1 + d.tc(a) / (ta - d.tc(a))), d.tm(b) * (1 + d.tm(a) / (ta - d.tm(a))))
        tas.append(ta * 1e6)
        tbs.append(tb * 1e6)
    return tas, tbs

def profile_single(
    k: Kernel,
    stream: torch.cuda.Stream = None,
    repeat = 100,
):
    for _ in range(3):
        k.run()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    if stream is None: 
        stream = torch.cuda.Stream(device = DEVICE)
    stream.wait_stream(torch.cuda.current_stream(DEVICE))
    
    g = torch.cuda.CUDAGraph()
    ts = []
    with torch.cuda.stream(stream):
        for _ in range(3): k.run()
        
        with torch.cuda.graph(g, stream=stream):
            for _ in range(repeat):
                k.run()
        for _ in range(100): g.replay()
        
        for _ in range(10):
            s.record(stream)
            g.replay()
            e.record(stream)
            e.synchronize()
            t = s.elapsed_time(e)
            ts.append(t * 1000 / repeat / k.cycle)
    g.reset()
    return float(np.median(ts))

def bench_with_bg_jobs_us(k1: Kernel, k2: Kernel,
                       s1: torch.cuda.Stream,
                       s2: torch.cuda.Stream,
                       repeat: int = 10,
                       num_iter: int = 100):
    if num_iter < 1 or repeat < 1:
        raise ValueError('num_iter and repeat must be positive')
    if s1.cuda_stream == s2.cuda_stream:
        raise ValueError('Foreground and background need different streams')
    from .environment import _cuda_call
    from cuda.bindings import driver
    external = driver.CUevent_record_flags.CU_EVENT_RECORD_EXTERNAL
    kernel, partner, stream, background = k1, k2, s1, s2
    # Initialize buffers and compile the selected candidate before capture.
    for job, owner in ((kernel, stream), (partner, background)):
        owner.wait_stream(torch.cuda.current_stream(DEVICE))
        with torch.cuda.stream(owner):
            for _ in range(3):
                job.run()
        owner.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    background_end = torch.cuda.Event(enable_timing=True)
    for event, owner in ((start, stream), (end, stream), (background_end, background)):
        event.record(owner)  # Materialize handles before capture.
    background_iter = num_iter
    for attempt in range(12):
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, stream=background):
                partner.run()
                stream.wait_stream(background)  # Partner starts before the timed window.
                with torch.cuda.stream(stream):
                    kernel.run()  # Exclude initial branch scheduling from the timing.
                    # Torch 2.7 needs the driver flag for replay timing events.
                    _cuda_call(driver.cuEventRecordWithFlags,
                                start.cuda_event, stream.cuda_stream, external)
                    for _ in range(num_iter):
                        kernel.run()
                    _cuda_call(driver.cuEventRecordWithFlags,
                                end.cuda_event, stream.cuda_stream, external)
                for _ in range(background_iter):
                    partner.run()
                _cuda_call(driver.cuEventRecordWithFlags,
                            background_end.cuda_event, background.cuda_stream, external)
                background.wait_stream(stream)
            samples = []
            with torch.cuda.stream(background):
                for idx in range(3 + repeat):
                    g.replay()
                    background.synchronize()
                    elapsed = start.elapsed_time(end)
                    if elapsed >= start.elapsed_time(background_end):
                        break  # Reject windows containing an uncontended tail.
                    if idx >= 3:
                        samples.append(elapsed)
        finally:
            g.reset()
        if len(samples) == repeat:
            break
        background_iter *= 2
    else:
        raise RuntimeError('Could not keep partner work queued for the timed window')
    return np.median(samples) * 1000 / num_iter / kernel.cycle

def maybe_tune_corun_single(k1: Kernel, 
                     k2: Kernel,
                     s1: torch.cuda.Stream,
                     s2: torch.cuda.Stream,
                     repeat: int = 10,
                     num_iter: int = 100):
    if k1.impl != 'triton-tuned': return
    best_config = None
    best_timing = float('inf')
    n_valid_configs = 0
    for config in k1.get_autotune_configs():
        k1.set_config(config)
        try:
            t = bench_with_bg_jobs_us(k1, k2, s1, s2, repeat, num_iter)
        except triton.OutOfResources:
            continue
        n_valid_configs += 1
        if t < best_timing:
            best_timing = t
            best_config = config
    if best_config is None:
        raise RuntimeError(f'No usable Triton configuration for {k1}')
    k1.set_config(best_config)
    print(f'Tuned {k1}: {n_valid_configs=} {s1.num_sms=},{s2.num_sms=},{best_config},{best_timing:.3f} us', flush=True)
    return best_config

def maybe_tune_corun(k1: Kernel, 
                     k2: Kernel,
                     s1: torch.cuda.Stream,
                     s2: torch.cuda.Stream,
                     max_steps: int = 5):
    for step in range(max_steps):
        maybe_tune_corun_single(k1,k2,s1,s2)
        maybe_tune_corun_single(k2,k1,s2,s1)

def profile_corun(
    a: Kernel,
    b: Kernel,
    num_iter = 100,
    repeat = 10,
    stream_mode: str = 'GC'
): 
    if num_iter < 1 or repeat < 1:
        raise ValueError('num_iter and repeat must be positive')
    device = DEVICE
    torch.cuda.set_device(device)
    total_sms = torch.cuda.get_device_properties(device).multi_processor_count
    from .profile_multi_stream import _get_partition_streams, _get_streams
    rows = []
    with ExitStack() as cleanup:
        if stream_mode == 'GC':
            streams = _get_partition_streams(total_sms, 2, 16, cleanup=cleanup)
        elif stream_mode == 'GC-b_only':
            a_stream = torch.cuda.Stream(DEVICE)
            a_stream.num_sms = total_sms
            streams = [_get_streams((sms,), cleanup) for sms in range(16, total_sms + 1, 16)]
            streams = [(a_stream, s[0]) for s in streams]
        elif stream_mode == 'GC-a_only':
            a_stream = torch.cuda.Stream(DEVICE)
            a_stream.num_sms = total_sms
            streams = [_get_streams((sms,), cleanup) for sms in range(16, total_sms + 1, 16)]
            streams = [(s[0], a_stream) for s in streams]
        elif stream_mode == 'MS':
            a_stream = torch.cuda.Stream(DEVICE)
            a_stream.num_sms = total_sms
            b_stream = torch.cuda.Stream(DEVICE)
            b_stream.num_sms = total_sms
            streams = [(a_stream, b_stream)]
        else:
            raise ValueError(f'Unknown stream_mode: {stream_mode}')
        print(f'{stream_mode}: {a.impl=},{b.impl=},{total_sms=},{len(streams)=}', flush=True)
        for sa, sb in streams:
            # warmup
            for _ in range(2):
                with torch.cuda.stream(sa):
                    a.run()
                with torch.cuda.stream(sb):
                    b.run()
            timings = []
            # Time each kernel while its partner has continuously queued work.
            maybe_tune_corun(a, b, sa, sb)
            for kernel, partner, stream, background in ((a, b, sa, sb), (b, a, sb, sa)):
                timings.append(bench_with_bg_jobs_us(kernel, partner, stream, background, repeat, num_iter))
            rows.append((sa.num_sms, sb.num_sms, *timings))
    return pd.DataFrame(rows, columns=['a_sms', 'b_sms', 'ta_us', 'tb_us'])

@dataclass 
class MLP(Kernel):
    M: int
    N: int
    K: int
    dtype: str
    _cycle: int = 1
    _impl: str = 'cublas' 
    _init: bool = False
    _config: triton.Config | None = None
    
    @property
    def f(self):
        return (self.M * self.N * self.K * 2) / 1e9 
    @property
    def m(self):
        return (self.M * self.N + self.M * self.K + self.N * self.K) * self._nelem(self.dtype) / 1e9
    @property
    def cycle(self):
        return self._cycle 

    @property
    def impl(self):
        return self._impl

    def get_autotune_configs(self):
        return matmul_kernel.configs

    def set_config(self, config):
        self._config = config

    def run(self):
        if not self._init:
            dtype = getattr(torch, self.dtype)
            self.A = [torch.randn(size = (self.M, self.K), dtype = dtype, device = DEVICE) for _ in range(self._cycle)]
            self.B = [torch.randn(size = (self.K, self.N), dtype = dtype, device = DEVICE) for _ in range(self._cycle)]
            self.C = [torch.empty(size = (self.M, self.N), dtype = dtype, device = DEVICE) for _ in range(self._cycle)]
            self._init = True
        for a,b,c in zip(self.A, self.B, self.C):
            if self._impl == 'cublas':
                torch.mm(a, b, out=c)
            elif self._impl in ('triton', 'triton-tuned'):
                grid = lambda META: ((triton.cdiv(self.M, META['BLOCK_SIZE_M'])) * triton.cdiv(self.N, META['BLOCK_SIZE_N']), )
                kernel = matmul_kernel if self._config is None else matmul_kernel.fn
                config = {} if self._config is None else self._config.all_kwargs()
                kernel[grid](
                    a, b, c,
                    self.M, self.N, self.K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    ACTIVATION='', **config
                )
    def __str__(self):
        return f'MLP({self.M}X{self.N}X{self.K})'

def plot(
    a: Kernel, 
    b: Kernel,
    d: Device,
    ax: plt.Axes
):
    # fig, ax = plt.subplots()
    tas, tbs = get_roofline(a,b,d, 10)
    
    roofline_color = 'blue'
    ax.vlines([d.t(a) * 1e6], min(tbs), max(tbs), linestyles = 'dashed', color = roofline_color)
    ax.hlines([d.t(b) * 1e6], min(tas), max(tas), linestyles = 'dashed', color = roofline_color)
    
    ax.plot(tas, tbs, marker = 'o', label = 'roofline', color = roofline_color)
    # ymax = max(tb_us, d.t(b) * 1e6)
    for _impl in ('cublas', 'triton', 'triton-tuned'):
        a._impl = b._impl = _impl
        ta_us = profile_single(a)
        tb_us = profile_single(b)
        for stream_mode in (
                            'GC', 
                            # 'GC-b_only',
                            # 'MS',
                            ):
            df = profile_corun(a, b, stream_mode = stream_mode)
            line, = ax.plot(df['ta_us'], df['tb_us'], marker = 'x', label = stream_mode + f'-{_impl}')
            if stream_mode == 'GC':
                ax.vlines([ta_us], df['tb_us'].min(), df['tb_us'].max(), linestyles = 'dashed', color = line.get_color())
                ax.hlines([tb_us], df['ta_us'].min(), df['ta_us'].max(), linestyles = 'dashed', color = line.get_color())
            # ymax = max(ymax, df['tb_us'].max())
            for idx, row in enumerate(df.itertuples(index=False)):
                ax.annotate(f'{row.a_sms}, {row.b_sms}',
                            (row.ta_us, row.tb_us), xytext=(5, 5 + 15 * (idx % 3)),
                            textcoords='offset points', fontsize=8,
                            arrowprops=dict(arrowstyle='-', color='0.5', lw=0.6))
    ax.set_ylim(0, tb_us * 3)
    ax.set_xlim(0, ta_us * 3)
    ax.set_xlabel(f'T {a} (us)')
    ax.set_ylabel(f'T {b} (us)')
    ax.grid()
    ax.legend()
    # fig.savefig(savepath)
    # print(f'{savepath=}')
def profile_encode_decode():     
    EncodeGEMMs = [
        # ('E1', (712, 16384, 2048)),
        # ('E2', (712, 2048, 16384)),
        ('E3', (712, 2560, 2048)),
        # ('E4', (712, 2048, 2048))
    ]    
    DecodeGEMMs = [
        ('D1', (10, 4096, 1024)),
        # ('D2', (10, 1024, 4096)),
        # ('D3', (10, 2560, 1024)),
        # ('D4', (10, 1024, 2048))
    ]
    fig, axes = plt.subplots(len(EncodeGEMMs), len(DecodeGEMMs),
                             figsize = (len(DecodeGEMMs) * 4, len(EncodeGEMMs) * 4))
    idx = 0
    for (E_name, E_shape), (D_name, D_shape) in itertools.product(EncodeGEMMs, DecodeGEMMs):
        a = MLP(*E_shape, dtype='float16')
        b = MLP(*D_shape, dtype='float16', _cycle = 10)
        d = Device("4090")
        if len(DecodeGEMMs) == 1 and len(EncodeGEMMs) == 1:
            ax = axes
        else:
            ax = axes[idx // len(DecodeGEMMs), idx % len(DecodeGEMMs)]
        idx += 1
        ax.set_title(f'{E_name} + {D_name}')
        print(f'Profiling {E_name} {E_shape} + {D_name} {D_shape}', flush=True)
        plot(a, b, d, ax)
        
    fig.tight_layout()
    fig.savefig('tmp.png')
    print('saved to tmp.png')
    

def debug_pair():
    from .profile_multi_stream import _get_streams
    '''
    Tuned MLP(712X2560X2048): n_valid_configs=12 s1.num_sms=80,s2.num_sms=48,BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 128, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 4, num_ctas: 1, num_stages: 4, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None,81.981 us
    Tuned MLP(10X4096X1024): n_valid_configs=12 s1.num_sms=48,s2.num_sms=80,BLOCK_SIZE_M: 16, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 2, num_ctas: 1, num_stages: 5, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None,11.742 us
    '''
    ka = MLP(712, 2560, 2048, dtype = 'float16', _impl = 'triton-tuned')
    kb = MLP(10, 4096, 1024, dtype = 'float16', _cycle = 10,
            _impl = 'triton-tuned')
    ca = triton.Config(kwargs={"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, **{"num_warps": 4, "num_ctas": 1, "num_stages": 4, "num_buffers_warp_spec": 0, "num_consumer_groups": 0, "reg_dec_producer": 0, "reg_inc_consumer": 0, "maxnreg": None})
    cb = triton.Config(kwargs = {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, **{"num_warps": 2, "num_ctas": 1, "num_stages": 5, "num_buffers_warp_spec": 0, "num_consumer_groups": 0, "reg_dec_producer": 0, "reg_inc_consumer": 0, "maxnreg": None
    })
    ka.set_config(ca)
    kb.set_config(cb)
    ta = profile_single(ka)
    tb = profile_single(kb)
    with ExitStack() as cleanup:
        sa, sb = _get_streams((112, 16), cleanup = cleanup)
        ta_alone = profile_single(ka, sa)
        tb_alone = profile_single(kb, sb)
        ta_corun = bench_with_bg_jobs_us(ka, kb, sa, sb)
        tb_corun = bench_with_bg_jobs_us(kb, ka, sb, sa)
    # TODO: for ncu profiling
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("E3_target"):
        ka.run()
    torch.cuda.synchronize()
    print(f'corun {sa.num_sms=},{sb.num_sms=},{ta=:.2f}us,{tb=:.2f}us,{ta_alone=:.2f}us,{tb_alone=:.2f},{ta_corun=:.2f}us,{tb_corun=:.2f}us')
if __name__ == '__main__':
    # profile_encode_decode()
    debug_pair()
