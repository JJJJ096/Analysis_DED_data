import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
from melt_pool_3D_plot★ import Data_load

data = Data_load()

def data_plot(data):
    plt.rcParams['font.family'] = 'Times New Roman' # 글꼴
    plt.rc('font', size=20)                         # 기본 폰트 크기
    plt.rc('axes', labelsize=30)                    # x,y축 label 폰트 크기
    plt.rc('xtick', labelsize=20)                   # x축 눈금 폰트 크기 
    plt.rc('ytick', labelsize=20)                   # y축 눈금 폰트 크기
    plt.rc('legend', fontsize=20)                   # 범례 폰트 크기
    plt.rc('figure', titlesize=50)                  # figure title 폰트 크기

    xlabel = np.linspace(0, len(data)*0.0001, len(data))
    xtick = np.linspace(0, len(data)*0.0001, 50)

    #fft = np.fft.fft(data['laser power'] / len(data['laser power']))
    #fft_magnitude = abs(fft)

    #signal_low = lowpassfiter(data['laser power'])
    #print(signal_low)

    ma_data = moving_average(data, 10)

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()
    line1 =ax1.plot(xlabel, data['melt pool temp'], color='r', linewidth=0.5, label='melt pool temp')
    ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("MeltPool Temperature(°C)")
    ax1.set_ylim([600, 2500])
    ax1.set_xticks(xtick)
    plt.yticks(fontsize=20)

    #line2 = ax2.plot(xlabel, data['laser power'], color='g', linewidth=0.1, label='laser power')
    line2 = ax2.plot(xlabel, ma_data, color='g', linewidth=0.3, label='laser power')
    ax2.set_ylabel("Laser Power")
    ax2.set_ylim([-100, 800])

def plot(data):
    plt.plot()