# TDMS File Handling Code
# Read TDMS, Synchronization, Interpolation

# TdmsFile.read(file_path)
# group = file['group_name']
# channel = group['channel_name'] or channel = tdms_file['group']['channel']
# time = channel.time_track().seconds
# channel.time_track(absolute_time = Ture)
# time을 기준으로 Synchroization,  HOW?
# data.interpolate(mathod='values') : x_axis, y_axis
# z_axis : Nan 채우기, HOW?  -> fillna(method = 'pad' and 'ffill')  fillna(method='bfill'or 'backfill') 데이터를 채우는 방향
# timestamp 2-64 seconds의 정밀도를 가짐
from nptdms import TdmsFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_csv_to_tdms():
    file_name = "E:/2021생기원/02.Monitoring/00.Labview 코드 개발/TDMS data/DED_monitoring.tdms"
    with TdmsFile.read(file_name, raw_timestamps=True) as tdms_file:
        tdms_file = TdmsFile.read(file_name, raw_timestamps=True)

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

    data.to_csv("test.csv")

def data_processing():
    file_name = "C:/Users/KAMIC/Downloads/202201061358_DED_Data.csv"
    data = pd.read_csv(file_name, header=0)
    data = data[['melt pool temp','x','y','z']]
    data.rename(columns={'melt pool temp':'temp'}, inplace=True)
    
    data = data.query('temp >= 1200')

    return data

def x_y_plane(data):
    print(data)
    resolution = 200
    step = 100 / resolution
    zero = np.zeros((resolution, resolution))
    zero_df = pd.DataFrame(zero)

    # layer_tickness = 0.25
    # layer_list = np.arange(0, max(data['z']), layer_tickness)

    cnt = 1
    for x in zero_df.index:
        for y in zero_df.columns:
            data.query('z == 8.5')
            t_data = data.query('{}< x <={} and {} < y <= {}'.format(x, x+1, y, y+1))
            
            if len(t_data) != 0:
                t = np.sum(t_data['temp']) / len(t_data)
                zero_df[x][y] = t
            if resolution^2 % cnt == 0:
                print("{} / {} ___ {}%".format(cnt, resolution^2, (cnt/resolution)*100))
            cnt+=1
            print(cnt)
    fig, ax = plt.subplots()

    im = ax.imshow(zero_df, cmap='jet',vmin=1000, vmax=2500)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Layer : {}'.format(8.5))
    # ax.set_ylim(0, 99)
    plt.tight_layout()
    plt.colorbar(im)
    plt.show()

def x_y_plane_resoultion():
    # parameters
    sample_rate = 200 # Hz
    travel_speed = 1000 # mm/min
    hatching_distance = 0.4 # mm

    # resolution calculate
    working_direction_aixs = round(((travel_speed/60) / sample_rate) * 6, 1)
    hatching_direction_axis = round(hatching_distance, 1)

    # zeros data frame
    x = np.arange(0, 100 + working_direction_aixs, working_direction_aixs)
    y = np.arange(0, 100 + hatching_direction_axis, hatching_direction_axis)
    from itertools import product
    zeros = pd.DataFrame(list(product(x, y)), columns=('x', 'y'))
    print(zeros)

    return zeros

def test_code(data):
    resolution = 1000
    t_data = data[data['z']==0.25]

    t_data['x_new'] = t_data['x'].round(1)
    t_data['y_new'] = t_data['y'].round(1)

    t_data_gridded = t_data['temp'].groupby([t_data['x_new'], t_data['y_new']]).mean().reset_index()

    x = np.arange(0, 100 , 0.1)
    y = np.arange(0, 100 , 0.1)
    from itertools import product
    zero_df = pd.DataFrame(list(product(x, y)), columns=('x', 'y'))

    zero_df = zero_df.merge(t_data_gridded, left_on=('x', 'y'), right_on=('x_new', 'y_new'), how = "left")
    zero_df = zero_df.fillna(0)
    
    pivot_table=zero_df.pivot_table(values='temp', index='x', columns='y')
    print(pivot_table)
    # print(zero_df)
    fig, ax = plt.subplots()

    im = ax.pcolor(pivot_table, cmap='jet',vmin=1000, vmax=1800)
    plt.show()


def x_y_plot(pivot_table):
    fig, ax = plt.subplots()

    im = ax.pcolor(pivot_table, cmap='jet',vmin=1000, vmax=1800)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Layer : {}'.format(8.5))
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.colorbar(im)
    plt.show()


if __name__ == '__main__':
    # data = data_processing()
    x_y_plane_resoultion()
