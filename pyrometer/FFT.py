from turtle import Turtle
import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import pandas as pd
from data_processing import focas

thr=True

def thr_fft(amplitude, thr):
    threshold = np.percentile(amplitude, thr)

    for i in range(len(amplitude)):
    	if amplitude[i] < threshold :
            Y[i] = 0 
    return Y

def inverse_fft(Y):
    ifft = np.fft.ifft(Y)
    return ifft

file_name = "C:/Users/KAMIC/Desktop/github/DED_monitoring/DAQ Labview code/2. Sennortherm(NIDAQ)/0927_data/0930_data_filter.xlsx"
data = pd.read_excel(file_name, sheet_name="DAQ",header=0)
data['melt pool temp'] = (data['melt pool temp'] - 2) * 200 + 700
data['laser power'] = data['laser power'] * 250
data["T"] = data['melt pool temp']
start, stop = 32000,42000
# start, stop = 10000,20000
data = data[start:stop]
# data = focas(800)

plt.hist(data['T'])
plt.show()

x = np.arange(0, len(data.index))
y = data["T"]
y = y.to_numpy()

Fs = 10000  # Sampling rate
T = 1/Fs      # Sample interval time
te = 1      # total time

NFFT = len(y)                           # 데이터 갯수
k = np.arange(NFFT)                     # 0:데이터갯수 1D array
f0 = k*Fs/NFFT                          # frequncy 정규화 
f0 = f0[range(math.trunc(NFFT/2))]      # math.trunc : 정수 부분만 반환 (소수점 제외), range(만큼 자름)
Y = np.fft.fft(y)                  # 양과 음의 주파수 크기를 계산 한 값
Y = Y[range(math.trunc(NFFT/2))]
amplitude = 2*abs(Y)                    # 계산된 주파수 영역의 값을 양의 값으로 변환하여 대칭인 점을 고려하여 2배 
phase_ang = np.angle(Y)*180/np.pi

if thr == True:
    Y = thr_fft(amplitude,99.5)

## 상위 10개의 주파수 출력
# idxy = np.argsort(-amplitude)
# for i in range(10):
#     print("freq={} amp={}".format(f0[idxy[i]], Y[idxy[i]]))

ifft = inverse_fft(Y)/2

# plot ##

fig = plt.figure()
plt.subplots_adjust(hspace = 0.4, wspace = 0.3)

ax1 = plt.subplot(311)
seires = ax1.plot(x/Fs, y, linewidth=0.4)
plt.title("Spectrum Analysis\n", fontsize=20)
ax1.set_xlabel("Time(s)")
ax1.set_ylabel("Temperature")
ax1.grid()

ax2 = plt.subplot(312)
FFT = ax2.plot(f0, amplitude, linewidth=1)
ax2.set_xlabel("Frequncy(Hz)")
ax2.set_ylabel("Amplitude")
ax2.grid()

ax3 = plt.subplot(313)

ax3.plot(range(0, len(ifft)), ifft, linewidth=1)
ax3.set_xlabel("time(s)")
ax3.set_ylabel("Temperautre")
ax3.grid()

plt.show()

