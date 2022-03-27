import math
from tkinter import X

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import statsmodels.api as sm
from nptdms import TdmsFile
from scipy.fftpack import fft, ifft

from data_processing import focas, save_csv_to_tdms

plt.rcParams['font.family'] = 'Times New Roman' # 글꼴
plt.rc('font', size=12)                         # 기본 폰트 크기
plt.rc('axes', labelsize=15)                    # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=12)                   # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=12)                   # y축 눈금 폰트 크기
plt.rc('legend', fontsize=12)                   # 범례 폰트 크기
plt.rc('figure', titlesize=20)                  # figure title 폰트 크기

def NFFT(x, y, Fs, te, thr=True, thr_value = 95):
    '''Nonequispaced Fast Fourier Transform 

        비 등간격 Fourier 변환을 통해 시계열 데이터를 time domain -> freqeuncy domain으로 변환하여 표현
        입력 받은 Fs(sample rate)의 절반 주파수인 F

        Parameters  :   x : pd.Series
                            time series data
                     
                        y : pd.Series
                            분석할 데이터
                     
                        Fs : int
                            Sampling Rate
                     
                        te : int
                            analysis time

        Returns     :   

        Requirment  :   pip install pandas
                        pip install numpy
                        pip install matplotlib
    '''
    # FFT 분석
    # window size = 0.05 s
    #       mp size = 0.8 mm, 
    #       Travel speed = 1000 mm/s = 16.667 mm/s
    #       window size = 0.05 s/mp
    #       sample rate * window size = 1.65 개
    #       100 period

    # offset 100, 165 개(5 s) 데이터 분석


    Fs = 10000  # Sampling rate
    T = 1/Fs      # Sample interval time
    te = 1      # total time

    n = len(y)
    NFFT = n
    k = np.arange(NFFT)
    f0 = k*Fs/NFFT                          # frequncy 정규화 
    f0 = f0[range(math.trunc(NFFT/2))]      # math.trunc : 정수 부분만 반환 (소수점 제외)
    Y = np.fft.fft(y)/NFFT                  # 양과 음의 주파수 크기를 계산 한 값
    Y = Y[range(math.trunc(NFFT/2))]
    amplitude = 2*abs(Y)                    # 계산된 주파수 영역의 값을 양의 값으로 변환하여 대칭인 점을 고려하여 2배 
    phase_ang = np.angle(Y)*180/np.pi
    print(amplitude)

    if thr == True:
         amplitude = thr_fft(amplitude, thr_value)

    ## 상위 10개의 주파수 출력
    idxy = np.argsort(-amplitude)
    for i in range(10):
        print("freq={} amp={}".format(f0[idxy[i]], Y[idxy[i]]))

    ifft = inverse_fft(Y)
    ## plot ##

    fig = plt.figure()
    plt.subplots_adjust(hspace = 0.4, wspace = 0.3)

    ax1 = plt.subplot(311)
    seires = ax1.plot(x/Fs, y, linewidth=0.4)
    plt.title("Spectrum Analysis\n", fontsize=20)
    ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("Temperature")
    ax1.grid()

    ax2 = plt.subplot(312)
    FFT = ax2.plot(f0, amplitude, linewidth=0.4) 
    # ax2.plot(f0, 2*Y) 
    ax2.set_xlabel("Frequncy(Hz)")
    ax2.set_ylabel("Amplitude")
    ax2.grid()

    ax3 = plt.subplot(313)
    ax3.plot(ifft, linewidth=0.4)
    ax3.set_xlabel("time(s)")
    ax3.set_ylabel("Temperautre")

    # phase = ax3.plot(f0, phase_ang, linewidth=0.4)
    # phase0= np.where(phase_ang==0)
    # phase90= np.where(phase_ang==90)
    # print(phase0)
    # print(phase90)
    # ax3.set_xlabel("Frequncy(Hz)")
    # ax3.set_ylabel('Phase')
    # ax3.grid()

    plt.show()


def thr_fft(amplitude, thr):
    threshold = np.percentile(amplitude, thr)
    print(threshold)
    for i in range(len(amplitude)):
    	if amplitude[i] < threshold :
            amplitude[i] = 0 
    return amplitude

def inverse_fft(Y):
    ifft = np.fft.ifft(Y)
    return ifft


def FFT(data, fs):
    df = 1 / fs
    x = np.arange(0, len(data.index))
    y = data["T"]

    nfft = len(x)
    df = fs/nfft
    k = np.arange(nfft)
    f = k * df
    nfft_half =math.trunc(nfft/2)
    f0 = f[range(nfft_half)]

    fft_y = np.fft.fft(y)/nfft *2
    fft_y0 = fft_y[range(nfft_half)]
    
    amp = abs(fft_y0)
    
    ## 상위 10개의 주파수 출력
    idxy = np.argsort(-amp)
    for i in range(10):
        print("freq={} amp={}".format(f0[idxy[i]], fft_y[idxy[i]]))

    ## 상위 20개 주파수 추출 및 합
    newy = np.zeros((nfft,)) 
    arfreq = [] 
    arcoec = [] 
    arcoes = [] 

    for i in range(20): 
        freq = f0[idxy[i]] 
        yx = fft_y[idxy[i]] 
        coec = yx.real 
        coes = yx.imag * -1 
        newy += coec * np.cos(2 * np.pi * freq * x) + coes * np.sin(2 * np.pi * freq * x) 
        arfreq.append(freq) 
        arcoec.append(coec) 
        arcoes.append(coes)

    # plt.figure() 
    # plt.plot(x, y, c='r', label='orginal') 
    # plt.plot(x, newy, c='b', label='fft') 
    # plt.legend() 
    # plt.show()


    ## 상위 12개 주파수를 하나씩 더해가며 데이터 복원
    # plt.figure(figsize=(20,15)) 
    # plti = 0 
    # ncnt = 11
    # newy = np.zeros((nfft,)) 

    # for i in range(ncnt+1): 
    #     freq = f0[idxy[i]] 
    #     yx = fft_y[idxy[i]] 
    #     coec = yx.real 
    #     coes = yx.imag * -1 
    #     newy += coec * np.cos(2 * np.pi * freq * x) + coes * np.sin(2 * np.pi * freq * x) 
        
    #     plti+=1 
    #     plt.subplot(4,4, plti) 
    #     plt.title("N={}".format(i+1)) 
    #     plt.plot(x, newy, label='fft')
    #     plt.plot(x, y, label='original')
    #     plt.legend()    
    #     # plt.plot(f0, amp)
    #     plt.xlabel("Frequbcy(Hz)")
    #     plt.ylabel("Amplitude")
    plt.show()

def line_slope(ss):
    X = np.arange(len(ss)).reshape(len(ss),1)
    linear.fit(X, ss)
    return linear.coef_

linear = sklearn.linear_model.LinearRegression()

def time_series_anlaysis(data, window_size):
    data['mean_t'] = data['T'].rolling(window_size).mean()
    data['max_t'] =  data['T'].rolling(window_size).max()
    data['min_t'] =  data['T'].rolling(window_size).min()
    data['slope'] =  data['T'].rolling(window_size).apply(line_slope)

    fig, axes = plt.subplots(4,1)
    plt.subplots_adjust(hspace = 0.4, wspace = 0.3)
    axes[0].plot(data.index/10000, data['T'], linewidth=0.4)
    axes[0].set_title("data predicte")
    axes[0].set_ylabel("origin data")
    axes[0].set_axis_off(True)

    axes[1].plot(data.index/10000, data['mean_t'], linewidth=0.4)
    axes[1].set_ylabel("Trand")
    axes[1].set_axis_off(True)

    axes[2].plot(data.index/10000, data['slope'], linewidth=0.4)
    axes[2].set_ylabel("slope")
    axes[2].set_axis_off(True)
    
    axes[3].plot()
    axes[3].set_ylabel("error")
    axes[3].set_xlabel("time(s)")
    plt.show()

def decompose(data, period):
    ''' time seires decompose
    
        statsmodel 을 이용하여 시계열데이터의 trend, seasonal, resid 등으로 분해

        Parameters : data : pd.DataFrame
                                time series data
                    period : 분석할 주파수
        Returns :

        Requirment : pip install pandas
                     pip install matplotlib
                     pip install statsmodels
    '''
    decomp = sm.tsa.seasonal_decompose(data['T'], period=period)
    
    fig, axes = plt.subplots(4,1)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
    axes[0].plot(data.index/10000, data['T'], linewidth=0.4)
    axes[0].set_title("Data Decompose\n")
    axes[0].set_ylabel("Original Data")
    # axes[0].get_xaxis().set_visible(False)

    axes[1].plot(data.index/10000, decomp.trend, linewidth=0.4)
    axes[1].set_ylabel("Trend")
    # axes[1].get_xaxis().set_visible(False)

    axes[2].plot(data.index/10000, decomp.seasonal, linewidth=0.4)
    axes[2].set_ylabel("Seasonal")
    # axes[2].get_xaxis().set_visible(False)
    
    axes[3].plot(data.index/10000, decomp.resid, linewidth=0.4)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("time(s)")
    plt.show()

def envelope(data, window_size=480, n=5, samepl_rate=10000):
    """ 포락선 분석
    """
    data['mean_t'] = data['T'].rolling(window_size).mean()

    envelope_up = data['mean_t'] + (data['mean_t'] * n/100)
    envelope_dn = data['mean_t'] - (data['mean_t'] * n/100)

    plt.plot(data.index/samepl_rate, data['T'], color='r', label="MP_temperature", linewidth=0.3)
    
    plt.plot(data.index/samepl_rate, data['mean_t'], color='b', label= "Moving average", linewidth=0.7)
    # , linestyle=(0, (5,5,1,5) ))
    plt.plot(data.index/samepl_rate, envelope_up, color='gray', label= "Envelop Up", linewidth=1)
    # , linestyle=(0, (5,5,1,5) ))
    plt.plot(data.index/samepl_rate, envelope_dn, color='gray', label= "Envelop Down", linewidth=1)
    # , linestyle=(0, (5, 10) ))
    plt.xlabel("time(s)")
    plt.ylabel("temperature('c)")
    plt.title("Melt Pool Temperauter")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # data = focas(thr_temp= 0)
    
    # file_name = "C:/Users/KAMIC/Desktop/github/DED_monitoring/DAQ Labview code/2. Sennortherm(NIDAQ)/0927_data/0930_data_filter.xlsx"
    # data = pd.read_excel(file_name, sheet_name="DAQ",header=0)
    # data['melt pool temp'] = (data['melt pool temp'] - 2) * 200 + 700
    # data['laser power'] = data['laser power'] * 250
    # data["T"] = data['melt pool temp']
    
    # start, stop = 32000,43000
    # # start, stop = 10000,20000
    # data = data[start:stop]
    data = focas(800)

    envelope(data, 99, 5, 10000)

    # decompose(data, 800)

    # x = np.arange(0, len(data.index))
    # y = data["T"]
    # # te = x/10000
    # NFFT(x, y, 10000, 1, True, 95)

    # time_series_anlaysis(data, 400)

    # FFT(data, fs=10000)