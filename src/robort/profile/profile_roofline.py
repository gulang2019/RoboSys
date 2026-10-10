import torch
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

class Hardware:
    name: str
    _gflops: dict[float, float]
    _membw: dict[float, float]

    def __init__(self, device_name, dtype = 'bf16'):
        csvs = {
            'NVIDIA GeForce RTX 4090': 'profile/rtx4090/hardware_profile.csv'
        }
        df = pd.read_csv(csvs.get(device_name, None))
        
        # rows = rows[[name.lower() in device_name.lower() for name in rows.Hardware]]
        # if len(rows) != 1:
            # raise ValueError(f'Expected one hardware row matching {device_name!r}')
        # row = rows.iloc[0]
        self.name = device_name
        self._membw = {row['sm_perc']:row['mem_bw_gbs'] for _, row in df.iterrows()}
        self._gflops = {row['sm_perc']:row['gflops_fp16'] for _, row in df.iterrows()}

    def gflops(self, sm_perc: float = 1.0):
        return self._gflops[sm_perc]
    
    def membw(self, sm_perc: float = 1.0):
        return self._membw[sm_perc]


class PerfModel:
    @dataclass(frozen=True)
    class Stage:
        aflops: float
        amem: float
        bmem: float
        
    _stages: dict[str, Stage]
    
    def __init__(self, device_name):
        csv = 'profile/stage.csv'
        self._stages = {row.Stage: self.Stage(float(row.aflops), float(row.amem), float(row.bmem))
                        for row in pd.read_csv(csv).itertuples(index=False)}

    def get_flops(self, s, bsz):
        s = self._stages[s]
        return s.aflops * bsz 
    
    
    def get_memacc(self, s, bsz):
        s = self._stages[s]
        return s.amem * bsz + s.bmem 
        

stages = ('embed', 'encode', 'decode')
schedules = [
    (('embed',), ('encode',), ('decode',)),
    (('encode',), ('embed', 'decode')),
    (('encode', 'decode'), ('embed',)),
    (('embed', 'encode', 'decode'),),
]

def _get_device_name():
    return torch.cuda.get_device_name(0)

def roofline_by_stage(batch_size, sm_percs):
    device_name = _get_device_name()
    print(device_name)
    perf_model = PerfModel(device_name)
    hardware = Hardware(device_name)
    measured = pd.read_csv('profile/profile_per_stage.csv')
    fig, axes = plt.subplots(1, len(stages), figsize = (len(stages) * 5, 5))
    for i, s in enumerate(stages):
        ax = axes[i] 
        rooflines = []
        t_flops = []
        t_mem = []
        m = measured[measured['stage'] == s][measured['batch_size'] == batch_size]
        for sm_perc in sm_percs:
            flops = perf_model.get_flops(s, bsz=batch_size)
            memacc = perf_model.get_memacc(s, bsz=batch_size)
            roofline = max(flops / hardware.gflops(sm_perc), memacc / hardware.membw(sm_perc)) * 1000
            t_flops.append(flops / hardware.gflops(sm_perc) * 1000)
            t_mem.append(memacc / hardware.membw(sm_perc) * 1000)
            rooflines.append(roofline)
        ax.plot(sm_percs, rooflines, label = 'roofline', marker = 'o')
        ax.plot(sm_percs, t_flops, label = 'flops', marker = 'x')
        ax.plot(sm_percs, t_mem, label = 'mem', marker = '|')
        ax.plot(m['sm_perc'], m['time_s'] * 1000, label = 'measured', marker = 'o')
        for line in (ax.lines[0], ax.lines[-1]):
            offset = 6 if line.get_label() == 'roofline' else -12
            for x, y in zip(line.get_xdata(), line.get_ydata()):
                ax.annotate(f'{y:.1f}', (x, y), xytext=(0, offset), textcoords='offset points',
                            ha='center', fontsize=7, color=line.get_color())
        ax.legend()
        ax.set_xlabel('sm percs')
        ax.set_ylabel('Latency (ms)')
        ax.set_yscale('log')
        ax.set_ylim(bottom=0)
        ax.grid(True)
        ax.set_title(s)
    
    
    fig.suptitle(f'bsz={batch_size}, {device_name}')
    fig.savefig('profile/roofline_by_stage.png')
    print('saved to profile/roofline_by_stage.png')
        

def roofline(batch_sizes):
    device_name = _get_device_name()
    perf_model = PerfModel(device_name)
    hardware = Hardware(device_name)
    print('exec on', hardware)
    data = []
    for bsz in batch_sizes:
        for sch in schedules:
            t = 0
            delay = 0
            _flops, _memacc = 0, 0
            for unit in sch:
                flops, memacc = 0, 0
                for s in unit:
                    flops += perf_model.get_flops(s, bsz)
                    memacc += perf_model.get_memacc(s, bsz)
                _flops += flops 
                _memacc += memacc
                # Ideal resource sharing: stages in this group finish together.
                _roofline_t = max(flops / hardware.gflops(), memacc / hardware.membw())
                delay += _roofline_t * len(unit)
                t += _roofline_t
            throughput = 1 / t * bsz
            data.append((device_name, bsz, sch, delay, throughput, _flops, _memacc))
    Path('profile').mkdir(exist_ok=True)
    df = pd.DataFrame(data, columns=['device', 'batch_size', 'schedule', 'delay', 'throughput', 'flops', 'memacc'])
    df.to_csv('profile/roofline.csv', index=False)
    
    fig_path = 'profile/tradeoffs_roofline.png'
    fig, ax = plt.subplots()
    for schedule, rows in df.groupby('schedule'):
        label = ' → '.join('+'.join(unit) for unit in schedule)
        ax.scatter(rows.delay * 1000, rows.throughput, label=label, s=12)
    frontier = df.sort_values(['delay', 'throughput'], ascending=[True, False]).drop_duplicates('delay')
    rates = frontier.throughput.round(10)
    frontier = frontier[rates > rates.cummax().shift(fill_value=-float('inf'))]
    ax.plot(frontier.delay * 1000, frontier.throughput, color='black', label='Pareto frontier')
    ax.set(xlabel='Delay (ms)', ylabel='Throughput (Inference/s)', title=f'{device_name} roofline')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return df

if __name__ == '__main__':
    # roofline(list(range(1,9)))
    roofline_by_stage(batch_size = 2, sm_percs = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625])
