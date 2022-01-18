from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import make_interp_spline

plt.rcParams['font.family'] = 'Times New Roman' # 글꼴
# plt.rc('font', size=20)                         # 기본 폰트 크기
# plt.rc('axes', labelsize=20)                    # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=20)                   # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=20)                   # y축 눈금 폰트 크기
# plt.rc('legend', fontsize=20)                   # 범례 폰트 크기
# plt.rc('figure', titlesize=30)                  # figure title 폰트 크기

def Data_load(): # data 불러오는 함수
    file_name = "C:/Users/KAMIC/Desktop/github/DED_monitoring/DAQ Labview code/2. Sennortherm(NIDAQ)/0927_data/0930_data_filter.xlsx"
    data = pd.read_excel(file_name, sheet_name="DAQ",header=0)
    return data

def data_processing(data):
    data['melt pool temp'] = (data['melt pool temp'] - 2) * 200 + 700
    data['laser power'] = data['laser power'] * 250
    return data

def moving_average(data, windowsize=120):
    laser_power = data['laser power'].rolling(windowsize).mean()
    return laser_power
   
def plot(data):
    xlabel = np.linspace(0, len(data)*0.0001, len(data))
    xtick = np.linspace(0, len(data)*0.0001, int((len(data)*0.01)/2))
    
    windowsize_1 = 10
    windowsize_2 = 40
    windowsize_3 = 70
    
    case_1 = moving_average(data, windowsize_1)
    case_2 = moving_average(data, windowsize_2)
    case_3 = moving_average(data, windowsize_3)
    
    print(case_1)
    print(case_2)
    print(case_3)

    print(len(xlabel))
    print(len(case_1))
    print(len(case_2))
    print(len(case_3))

    fig, ax = plt.subplots(4,1,figsize=(16,9))
    ax1 = ax[0].twinx()
    ax2 = ax[1].twinx()
    ax3 = ax[2].twinx()
    ax4 = ax[3].twinx()
    print(np.shape(xlabel))
    print(np.shape(data['laser power']))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    ax1.plot(xlabel, data['laser power'], label='origin', linewidth=0.5, color='orange')
    ax[0].plot(xlabel, data['melt pool temp'], linewidth=0.5, color='r')
    ax2.plot(xlabel, case_1, linewidth=0.5, color='b')
    ax[1].plot(xlabel, data['melt pool temp'], linewidth=0.5, color='r')
    ax3.plot(xlabel, case_2, linewidth=0.5, color='g')
    ax[2].plot(xlabel, data['melt pool temp'], linewidth=0.5, color='r')
    ax4.plot(xlabel, case_3, linewidth=0.5, color='black')
    ax[3].plot(xlabel, data['melt pool temp'], linewidth=0.5, color='r')

    ax[0].set_title("original")
    ax[1].set_title("windowsize_{}".format(windowsize_1))
    ax[2].set_title("windowsize_{}".format(windowsize_2))
    ax[3].set_title("windowsize_{}".format(windowsize_3))
    
    ax[3].set_xlabel("time(s)", fontsize=20)
    fig.text(0.05, 0.5, "melt pool temperature(°C)", va='center', rotation='vertical', fontsize=20)
    fig.text(0.95, 0.5, "Laser power(K)", va='center', rotation='vertical', fontsize=20)

    #plt.tight_layout()
    plt.show()

def plotly(data):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    
    xlabel = np.linspace(0, len(data)*0.0001, len(data))
    xtick = np.linspace(0, len(data)*0.0001, int((len(data)*0.01)/2))
    
    windowsize_1 = 10
    windowsize_2 = 40
    windowsize_3 = 70
    
    case_1 = moving_average(data, windowsize_1)
    case_2 = moving_average(data, windowsize_2)
    case_3 = moving_average(data, windowsize_3)
    

    fig = make_subplots(rows=4, cols=1)

    fig.add_trace(px.line(x=xlabel, y=data['laser power']))
    
    fig.show()

def single_plot(data):
    #spline_1 = make_interp_spline(data.index, moving_average(data))

    fig, ax = plt.subplots(figsize=(12,8))
    ax1 = ax.twinx()
    
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    p1 = ax.plot(data.index*0.0001, data['melt pool temp'], linewidth=0.7, color='r', label="Melt pool temp")
    ax.set_title("Melt pool temperature VS Laser power\n", fontsize=30)
    ax.set_ylabel("Melt pool temperature(℃)", fontsize=30)
    ax.set_xlabel("time(s)", fontsize=30)

    # ax.set_xticks(fontsize=20)
    # ax.set_yticks(fontsize=20)
    #p2 = ax1.plot(data.index*0.0001, spline_1, linewidth=0.7, color='g', label='Laser power')
    p2 = ax1.plot(data.index*0.0001, moving_average(data), linewidth=0.7, color='g', label='Laser power')
    print(p1)
    ax1.set_ylabel("Laser power(K)", fontsize=30)
    ax1.set_ylim([-50, 700])
    plt.yticks(fontsize=20)
    leg =p1 + p2
    ax.legend(leg, ['Melt pool temp', 'Laser power'], loc='upper right', fontsize=15)

    plt.tight_layout()
    plt.show()

def main():
    data = Data_load()
    data = data_processing(data)
    single_plot(data)
    #plotly(data)

if __name__ == '__main__':
    main()