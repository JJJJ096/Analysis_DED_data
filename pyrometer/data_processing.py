import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from nptdms import TdmsFile

# focas(.txt) file load
def focas(thr_temp=1200):
    """ Load focas data file
        
        DED에서 수집한 포카스 파일을 불러와 pd.DataFrame 형태로 반환하는 함수
        tkinter를 이용하여 .txt 파일을 open 한다.

        Parameters: thr_temp: (defalt: 1200)
                        thr_temp 이상의 데이터를 필터링하여 가져온다.
                        이는 laser off(700도 이하) 상태의 테이터를 필터링 하기 위해 사용
        
        Returns:    data: pd.DataFrame
                        col_name = ['x','y','z','c','a','T']의 열을 가진다.
                        불러온 데이터를 thr_temp 값으로 필터링하여 data로 반환
                    
                    print:
                        데이터의 앞 부분 5개 행.
                        데이터 describe를 출력.
        
        Requirment  pip install python3-tk 
                    pip install pandas

    """
    file_name = fd.askopenfilename(initialdir='C:/User',
                                    title="file", filetypes=(("txt files", "*.txt"),("all files", "*.*")))
    data = pd.read_csv(file_name, header=None, names=['date', 'num', 'x','y','z','c','a','vol','T'])
  
    data = data[['x','y','z','c','a','T']]
    data = data.query('T >= {}'.format(thr_temp))
    # data = data.query('2.55 <= z ')
    data = data.reset_index()
    print("\nData header \n{}".format(data.head(5)))
    print("\nnfocas data describe\n{}".format(data.describe()))
    return data

# TDMs file load
def TDMs(file_name, thr_temp=1200):
    """ Load TMDs data file
        
        DED에서 수집한 TDMs 파일을 불러와 pd.DataFrame 형태로 반환하는 함수
        
        Parameters: file_name : file_name.csv
                        불러올 파일 경로를 입력한다.
                        전처리 된 csv 파일을 불러와야 함
        
                    thr_temp: (defalt: 1200)
                        thr_temp 이상의 데이터를 필터링하여 가져온다.
                        이는 laser off(700도 이하) 상태의 테이터를 필터링 하기 위해 사용
            
        Returns:    data: pd.DataFrame
                        col_name = ['x','y','z','c','a','T']의 열을 가진다.
                        불러온 데이터를 thr_temp 값으로 필터링하여 data로 반환
        
        Requirment  pip install pandas

        Example:
                    >>>file_name = "TDMs_file_path.csv"
                    >>>data = TDMs(file_name, 1200)
    """
    # file_name = = "C:/Users/KAMIC/Downloads/202201061358_DED_Data.csv"
    data = pd.read_csv(file_name, header=0)
    data = data[["time", "melt pool temp", "laser power", 'x','y','z','c', 'a']]
    data.rename(columns={'melt pool temp':'T'}, inplace=True)
    data = data.query('T >= {}'.format(thr_temp))
    data = data.reset_index()
    # print(data.describe())
    return data

# Labview에서 수집한 TDMs  -> csv
def save_csv_to_tdms(load_file, save_file_name=None):
    """ TDMs data file processing
        
        DED에서 수집한 TDMs 파일의 각 raw 데이터를 시간 값으로 동기화
        가장 빠른 DAQ 데이터의 수집속도에 맞춰 다른 데이터들을 선형 보간한다.
        처리된 데이터를 csv file로 저장 및 pd.DataFrame 형태로 반환하는 함수
        
        Parameters: load_file :
                        불러올 TDMs 파일 경로를 입력한다.
        
                    save_file_name: file_name.csv
                        저장 할 파일 명을 입력 받아 저장한다.

        Returns:    data: pd.DataFrame
                        col_name = ['x','y','z','c','a','T']의 열을 가진다.
        
        Requirment  pip install pandas
                    pip install npTDMS

        Example:
                    >>>load_file = "TDMs_file_path.tdms"
                    >>>save_file_name = "save_file_name.csv"
                    >>>data = save_csv_to_tdms(load_file, save_file_name)
                    >>>(save_file_name.csv)
    """
    # file_name = "E:/2021생기원/02.Monitoring/00.Labview 코드 개발/TDMS data/DED_monitoring.tdms"
    load_file = "C:/Users/KAMIC/Downloads/20220127101335.tdms"
    with TdmsFile.read(load_file, raw_timestamps=True) as tdms_file:
        tdms_file = TdmsFile.read(load_file, raw_timestamps=True)

        mpt = tdms_file['DAQ']['Melt pool temp']
        lp = tdms_file['DAQ']['Laser power']
        x_axis = tdms_file['Position']['x axis']
        y_axis = tdms_file['Position']['y axis']
        z_axis = tdms_file['Position']['z axis']
        c_axis = tdms_file['Position']['c axis']
        a_axis = tdms_file['Position']['a axis']

    time_daq = mpt.properties['wf_start_time'].seconds + mpt.time_track()
    time_position = x_axis.properties['wf_start_time'].seconds + x_axis.time_track()

    if (len(mpt)-len(lp)) != 0:
        time_daq = time_daq[0:-(abs(len(mpt)-len(lp)))]
        if len(mpt) > len(lp):
            mpt = mpt[0:-(len(mpt)-len(lp))]
        elif len(mpt) < len(lp):
            lp = lp[0:(len(lp)-len(mpt))]

    print("mpt: {}, lp : {}, time_daq :{}".format(len(mpt), len(lp), len(time_daq)))
    # DAQ_time = mpt.time_track(absolute_time=True)
    DAQ_df = pd.DataFrame(data={'time':time_daq, "melt pool temp":mpt, "laser power":lp})        
    # position_time = x_axis.time_track(absolute_time=True)
    position_df = pd.DataFrame(data={'time':time_position, 
                                'x':x_axis, 
                                'y':y_axis, 
                                'z':z_axis, 
                                'c':c_axis, 
                                'a':a_axis})
    data = pd.merge(DAQ_df, position_df, how='outer',on='time')
                                                            #, indicator=True) # time을 기준으로 데이터를 합침
    #data = data.interpolate(method='values') # imterpolation
    # fillna(method = 'pad' and 'ffill')  fillna(method='bfill'or 'backfill') 데이터를 채우는 방향
    data['x'] = data['x'].interpolate(method='values')
    data['y'] = data['y'].interpolate(method='values')
    data['z'] = data['z'].fillna(method='ffill')
    data['c'] = data['c'].interpolate(method='values')
    data['a'] = data['a'].fillna(method='ffill')
    # data['melt pool temp'] = (data['melt pool temp'] -2)*200 + 700
    # data['laser power'] = data['laser power'] * 250
    # print(data)

    data.to_csv("{}.csv".format(save_file_name))
    return data

# 5축 데이터를 3축으로 변환하는 함수
def data_transform(x,y,z,c,a):
    """ 5 axis data => 3 axis data
        
        5축 데이터를 3축 데이터로 변환하여 x,y,z 값을 반환 하는 함수
    
        Parameters: x : pd.Series

                    y : pd.Series

                    z : pd.Series

                    c : pd.Series

                    a : pd.Series

        Returns:    xx: pd.Series
                        변환 된 x 값을 반환
                    yy: pd.Series
                        변환 된 y 값을 반환
                    zz: pd.Series
                        변환 된 z 값을 반환
        
        Requirment:  pip install pandas
                     pip install numpy
        
        Example:
                    >>>data = focas()
                    >>>x, y, z = data_transform(data.x, data.y, data.z, data.c, data.a, data.T)
    """
    pi = np.pi
    c_rad = c / 180 * pi
    a_rad = a / 180 * pi
    xx = x * np.cos(c_rad) * np.cos(a_rad) - y * np.sin(c_rad) + z * np.cos(c_rad) * np.sin(a_rad)
    yy = x * np.sin(c_rad) * np.cos(a_rad) + y * np.cos(c_rad) + z * np.sin(c_rad) * np.sin(a_rad)
    zz = - x * np.sin(a_rad) + z * np.cos(a_rad)
    return xx, yy, zz

def temperature_per_layer(data, height, layer_thickness=0.25):
    """ Average of layer temperature 
        
        z layer에 따른 온도 평균을 계산하고 반환 하는 함수
    
        Parameters: data: pd.DataFrame
                        focas or TDMs data
                        data must be include row(x,y,z,c,a,T)
                    height: float
                        데이터의 총 높이
                    layer_thickness: (defalt : 0.25)
                        적층 시 레이어 두께 입력 

        Returns:    layer_temp_mean: list
                        layer 당 평균 온도를 list로 반환
                    layer_temp_max: float
                        layer의 최고 온도를 반환
                    layer_list: list
                        layer thickness에 따른 z list를 반환
        
        Requirment:  pip install pandas
                     pip install numpy
        
        Example:
                    >>>data = focas()
                    >>>mean, max, layer = temperature_per_layer(data, 10)
    """    
    layer_temp_mean = []
    layer_temp_max = []
    layer_list = np.arange(0, height + layer_thickness, layer_thickness)
    layer_list = np.round(layer_list, 3)
    print("/nz layer list(mm) : {}".format(layer_list))
    for z in layer_list:
        layer_temp = data.query('z == {}'.format(z))
        layer_mean = np.mean(layer_temp['T'])
        layer_max = np.max(layer_temp['T'])
        layer_temp_mean.append(round(layer_mean, 3))
        layer_temp_max.append(round(layer_max,3))
    print("/n z layer average of temperature".format(layer_temp_mean))
    return layer_temp_mean, layer_temp_max, layer_list

def moving_average(data, windowsize=120, col_name=None):
    """ Calibration data moving average 
        
        데이터의 노이즈 제거를 위해 이동평균을 계산하여 데이터 반환
        이동평균(moving average) : 평균을 구하되 구하고자 하는 전체 데이터에 대한 평균이 아니라,
        전체 데이터의 일부분씩 순차적으로 평균을 구함
    
        Parameters: data: pd.DataFrame
                        focas or TDMs data
                        data must be include row(x,y,z,c,a,T)
                    windowsize: (defalt : 120)
                        windowsize 만큼의 데이터를 묶어 계산
                        ex) 총 데이터 갯수 100개, winowsize 10개 -> 99개의 데이터 반환
                    col_name: str, "column_name"
                        이동평균을 계산 할 데이터의 열 이름

        Returns:    moving_average: pd.Series
                        계산 된 이동 평균 값을 반환
        
        Requirment:  pip install pandas
        
        Example:
                    >>>data = focas()
                    >>>movig_average = moving_average(data, 100, "laser power")
    """        
    moving_average = data['{}'.format(col_name)].rolling(windowsize).mean()
    return moving_average

def lowpassfiter(signal, thresh = 0.63, wavelet='db4'):
    """ lowpassfiter data

            사용자가 지정한 특정 값을 유지하는 차단 주파수 보다 낮은 주파수를 갖는 신호를 통과 시켜 필터링 하는 함수

        Parameters: signal: pd.Series
                        lowpassfilter를 적용할 데이터
                    thresh: (defalt : 0.63)
                    
                    wavelet:
                        wavelet 적용 방식

        Returns:    moving_average: pd.Series
                        계산 된 이동 평균 값을 반환
        
        Requirment: pip install pandas
                    pip install numpy
                    pip install PyWavelets
        
        Example:

    """            
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signanl = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signanl

def FFT(data, col_name=None):
    """ Fast Fourier Transform

            고속 푸리에 변환 알고리즘을 계산하는 함수

        Parameters: data: pd.DataFrame
                        focas or TDMs data
                        data must be include row(x,y,z,c,a,T)
                    col_name: str, "column_name"
                        푸리에 변환을 적용 할 데이터의 열 이름

        Returns:    fft_magnitude: pd.Series
                        계산 값을 반환
        
        Requirment: pip install pandas
                    pip install numpy
        
        Example:

    """                

    Fs = 33
    L = len(data['{}'.format(col_name)])

    fft = np.fft.fft(data['{}'.format(col_name)]) / len(data['{}'.format(col_name)])
    #fft = 10000/ len(data['laser power'])
    fft_magnitude = abs(fft)
    return fft_magnitude
