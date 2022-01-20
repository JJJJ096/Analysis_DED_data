import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from nptdms import TdmsFile

# focas(.txt) file load
def focas(thr_temp=1200):
    file_name = fd.askopenfilename(initialdir='C:/User',
                                    title="file", filetypes=(("txt files", "*.txt"),("all files", "*.*")))
    data = pd.read_csv(file_name, header=None, names=['date', 'num', 'x','y','z','c','a','vol','T'])
  
    data = data[['x','y','z','c','a','T']]
    data = data.query('T >= {}'.format(thr_temp))
    # data = data.query('2.55 <= z ')
    data = data.reset_index()
    
    return data

# TDMs file load
def TDMs(file_name, thr_temp=1200):
    # file_name = = "C:/Users/KAMIC/Downloads/202201061358_DED_Data.csv"
    data = pd.read_csv(file_name, header=0)
    data = data[['melt pool temp','x','y','z']]
    data.rename(columns={'melt pool temp':'temp'}, inplace=True)
    data = data.query('temp >= {}'.format(thr_temp))

    return data

# Labview에서 수집한 TDMs  -> csv
def save_csv_to_tdms(load_file, save_file_name):
    # file_name = "E:/2021생기원/02.Monitoring/00.Labview 코드 개발/TDMS data/DED_monitoring.tdms"
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

    # DAQ_time = mpt.time_track(absolute_time=True)
    DAQ_df = pd.DataFrame(data={'time':time_daq,
                                "melt pool temp":mpt, 
                                "laser power":lp})
        
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

    data['melt pool temp'] = (data['melt pool temp'] -2)*200 + 700
    data['laser power'] = data['laser power'] * 250
    # print(data)

    data.to_csv("{}.csv".format(save_file_name))

# 5축 데이터를 3축으로 변환하는 함수
def data_transform(x,y,z,c,a):
    pi = np.pi
    c_rad = c / 180 * pi
    a_rad = a / 180 * pi
    xx = x * np.cos(c_rad) * np.cos(a_rad) - y * np.sin(c_rad) + z * np.cos(c_rad) * np.sin(a_rad)
    yy = x * np.sin(c_rad) * np.cos(a_rad) + y * np.cos(c_rad) + z * np.sin(c_rad) * np.sin(a_rad)
    zz = - x * np.sin(a_rad) + z * np.cos(a_rad)
    return xx, yy, zz

def temperature_per_layer(data, height, layer_thickness=0.25):
    
    layer_temp_mean = []
    layer_temp_max = []
    layer_list = np.arange(0, height, layer_thickness)
    layer_list = np.round(layer_list, 3)
    print(layer_list)
    for z in layer_list:
        layer_temp = data.query('z == {}'.format(z))
        layer_mean = np.mean(layer_temp['T'])
        layer_max = np.max(layer_temp['T'])
        layer_temp_mean.append(round(layer_mean, 3))
        layer_temp_max.append(round(layer_max,3))
    print(layer_temp_mean)
    return layer_temp_mean, layer_temp_max

