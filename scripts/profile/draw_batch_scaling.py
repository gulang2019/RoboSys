import pandas as pd
import matplotlib.pyplot as plt

paths = (
    ('RTX Thor', 'profile/thor_flashrt/policy_profile.csv'),
    ('RTX 4090', 'profile/rtx4090_flashrt/policy_profile.csv'),
    ('RTX 5090', 'profile/rtx5090_flashrt/policy_profile.csv'),
)

def draw(ax, hardware, path):
    df = pd.read_csv(path)
    num_views =  2
    df = df[(df['num_views'] == num_views) & (df['backend'] == ('flash_rt'))].copy()
    df = df.sort_values('batch_size')
    df['hardware'] = hardware
    df.loc[df['stage'] == 'decode', 'lat_mean'] *= 10 / df['num_steps'].iloc[0]

    total = df.groupby('batch_size', as_index=False)['lat_mean'].sum()
    total['stage'], total['hardware'] = 'total', hardware
    df = pd.concat([df, total], ignore_index=True)
    
    for stage_name, stage_df in df.groupby('stage'):
        line, = ax.plot(stage_df['batch_size'], stage_df['lat_mean'], marker = 'o', label = stage_name)
        for batch_size, latency in zip(stage_df['batch_size'], stage_df['lat_mean']):
            offset = -14 if stage_name == 'embed' else 8
            ax.annotate(
                f'({batch_size}, {latency:.2f})',
                xy=(batch_size, latency), xytext=(0, offset),
                textcoords='offset points', ha='center', fontsize=10,
                color='black',
            )

    ax.legend()
    ax.set_xlabel('batch size')
    ax.set_ylabel('latency (ms)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.margins(x=0.15, y=0.2)
    ax.set_title(hardware)
    
    return df[['hardware', 'stage', 'batch_size', 'lat_mean']]

fig, axes = plt.subplots(1, len(paths), figsize = (4 * len(paths), 4), layout='constrained')
fig.suptitle('Point labels: (batch size, latency in ms); decode latency ×10')
latency_rows = []
for ax, (hardware_name, path) in zip(axes.flatten(), paths):
    latency_rows.append(draw(ax, hardware_name, path))
latencies = pd.concat(latency_rows, ignore_index=True).pivot(
    index=['hardware', 'stage'], columns='batch_size', values='lat_mean'
).sort_index(axis=1)
latencies.to_csv('batch_scaling.csv')
fig.savefig('batch_scaling.png')
