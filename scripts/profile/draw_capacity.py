csv_file = 'profile/rtx4090/hardware_profile.csv'
plot_path = 'profile/hardware_trend.png'

import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv(csv_file)
df = df.sort_values('sm_perc')
fig, ax = plt.subplots()
df.plot(ax=ax, x = 'sm_perc', y = 'mem_bw_gbs', color = 'red', marker = 'x')
df.plot(ax=ax.twinx(), x = 'sm_perc', y = 'gflops_fp16', marker = 'o')
ax.set_xlabel('sm perc')
ax.set_ylabel('mem_bw')
ax.twinx().set_ylabel('gflops')
ax.twinx().set_ylim(bottom = 0)
fig.savefig(plot_path)
print(f'saved to {plot_path}')