import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from pyrometer.data_processing import focas, TDMs, save_csv_to_tdms, data_transform, temperature_per_layer

data = focas()
print(data.describe())

fig= plt.figure(figsize=(10,10))

ax1 = plt.subplot(212)
series = ax1.scatter(data.index, data['T'], c='r', s=0.5, label='series')

mean_plot_data_y = [data['T'].mean(),data['T'].mean()]
mean_plot_data_x = [data.index[0], data.index[-1]]
mean = ax1.plot(mean_plot_data_x, mean_plot_data_y, linewidth=3, linestyle='-.', color='g', label='mean')
ax1.set_xlabel('time')
ax1.set_ylabel('temperature')
ax1.set_title('data analysis')
ax1.legend()

ax2 = plt.subplot(221)
hist = ax2.hist(data['T'], bins=100, label='hist', color='gray', rwidth=0.5)
ax2.set_xlabel('temp')
ax2.set_ylabel('count')
ax2.set_title('histogram')
ax2.legend()

ax3 = plt. subplot(222)
rv = sp.stats.norm(loc=data['T'].mean(), scale=data['T'].std())
analysis = data.sort_values(by=['T'], axis=0)
pdf = ax3.plot(analysis['T'], rv.pdf(analysis['T']), color='r')
ax3.set_xlabel('temperature')
ax3.set_ylabel(' ')
ax3.set_title('probability density function')

plt.tight_layout()
plt.show()

