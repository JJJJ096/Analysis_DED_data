from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
import pywt

def Data_load(): # data 불러오는 함수
    file_name = "C:/Users/KAMIC/Desktop/github/DED_monitoring/DAQ Labview code/2. Sennortherm(NIDAQ)/0927_data/0930_data_filter.xlsx"
    data = pd.read_excel(file_name, sheet_name="DAQ",header=0)
    return data

def data_processing(data):
    data['melt pool temp'] = (data['melt pool temp'] - 2) * 200 + 700
    data['laser power'] = data['laser power'] * 250

    return data 

def lowpassfiter(signal, thresh = 0.63, wavelet='db4'):
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signanl = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signanl

def moving_average(data, interval=100):
    return data.rolling(interval).mean()

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


    #ax3 = ax1.twinx()
    #line3 = ax3.plot(xlabel, signal_low,color='blue', linewidth=1, label='wavelet')
    #ax3.set_ylim([-100, 800])

    #lines = line1 + line2 + line3
    lines = line1 + line2
    #labels = ['melt pool temp', 'laser power', 'wavelet']
    labels = ['melt pool temp', 'laser power']
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title("Melt pool temp VS Laser power\n")
    plt.tight_layout()
    plt.show()

def FFT(data):
    Ts = 10000
    Fs = 1/Ts
    L = len(data['laser power'])

    fft = np.fft.fft(data['laser power']) / len(data['laser power'])
    #fft = 10000/ len(data['laser power'])
    fft_magnitude = abs(fft)

    fig, ax1 = plt.subplots(figsize=(16,9))
    line1 =ax1.plot(fft_magnitude, color='r', linewidth=0.2)
    ax1.set_xlabel("Frequency", fontsize=15)
    ax1.set_ylabel("Laser power", fontsize=20)

    plt.tight_layout()
    plt.show()

def main():
    data = Data_load()
    data = data_processing(data)
    data_plot(data)
    #FFT(data)

    #print(data)
    #pp = (data['melt pool temp'][228712] - data['melt pool temp'][229505]+273) / (22.8712-22.9505)
    #print(pp)
    #index = [228768, 228871,229200]
    #cooling_rate = []
    #for i in index:
    #    print('temperature : {}'.format(data['melt pool temp'][i]))
    #    point = ((abs(data['melt pool temp'][i] - data['melt pool temp'][i+300]))+273) / (300*0.0001)
    #    cooling_rate.append(point)
    #print(cooling_rate)

if __name__ == "__main__":
    
    main()