from random import sample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from data_processing import focas, TDMs, save_csv_to_tdms, data_transform, temperature_per_layer

plt.rcParams['font.family'] = 'Times New Roman' # 글꼴
plt.rc('font', size=12)                         # 기본 폰트 크기
plt.rc('axes', labelsize=15)                    # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=12)                   # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=12)                   # y축 눈금 폰트 크기
plt.rc('legend', fontsize=12)                   # 범례 폰트 크기
plt.rc('figure', titlesize=20)                  # figure title 폰트 크기

def meltpool_temperature_spatial_3d_map(x, y, z, T, vmin=1200, vmax=2300, zmax=100, zmin=0):


    fig = plt.figure(figsize=(12,10))
    ax = plt.gca(projection='3d')
    sctt = ax.scatter(x, y, z, alpha=1, c=T, cmap='jet', s=20, vmin=vmin, vmax=vmax, lw=0)
    plt.title("Melt Pool Temperature spatial 3d map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    #ax.grid(False)
    #ax.set_axis_off()
    # ax.set_frame_on(True)
    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 100)
    ax.set_zlim(zmin, zmax)
    fig.colorbar(sctt, ax=ax, shrink=0.7, aspect=10)
    ax.view_init(20, 90)                                        # 그래프 각도 view_init(z축, x-y축)
    plt.tight_layout()
    plt.show()

def layer_bar(data, temp_min=1200, temp_max=2100):
    layer_temp_mean, layer_temp_max = temperature_per_layer(data, 3)

    width = np.arange(0, 4.75, 0.15)
    # width = width.tolist()
    layer_temp_mean = np.asarray(layer_temp_mean)
    layer_temp_max = np.asarray(layer_temp_max)

    xerr = layer_temp_max-layer_temp_mean
    yerr = np.zeros(len(xerr))
    error = [yerr,xerr]

    # fig = plt.figure(figsize=(12, 10))

    plt.barh(width, layer_temp_mean, height=0.1, color='r')
    # , xerr=error, capsize=2, linewidth=0.1)
    plt.xlabel("temperature(℃)")
    plt.ylabel("layer(mm)")
    plt.xlim(1200, 2100)
    plt.title("melt pool temperature per layer")
    plt.show()

def time_series(data, sample_rate=33):

    fig = plt.figure(figsize=(12,8))
    
    plt.plot(data['T'].index/sample_rate, data['T'], color='r', linewidth='0.4')
    # plt.scatter(t.index/33, t, color='r', s=0.2)
    #plt.scatter(t.index/33, t, s=1)
    plt.xlabel('time(s)', fontsize=20)
    plt.ylabel('Temperature(℃)', fontsize=20)
    plt.title('Melt pool temperature\n', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.show()

def x_y_2D_scatter(data, z):
    data=data.query('z=={}'.format(z))
    # print(data)

    fig, ax = plt.subplots()
    layer = ax.scatter(data.x, data.y, c=data['T'], marker='s', cmap='jet')
    ax.set_ylim(0,100)
    ax.set_xlim(0,100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('layer')
    plt.tight_layout()
    plt.colorbar(layer)
    plt.show()

def pdf_plot(data):
    import scipy as sp
    import scipy.stats
    rv = sp.stats.norm(loc=data['T'].mean(), scale=data['T'].std())
    analysis = data.sort_values(by=['T'], axis=0)
    
    pdf = plt.plot(analysis['T'], rv.pdf(analysis['T']), color='r')
    plt.xlabel('temperature')
    plt.ylabel(' ')
    plt.title('probability density function')
    plt.show()

def visualization_data(data, layer_thickness=0.25, sample_rate=33, height=0.75):
    """DED monitoring data visualization
    
        DED 모니터링 데이터를 시각화 하기 위한 코드
        time series, histogram, layer average temperature, pdf, x-y plane 그래프를 나타낸다.

        Parameters: data: pd.DataFrame
                            focas or TDMs data, data must be include row(x,y,z,c,a,T)

                        layer_thickness : = 0.25
                            데이터의 layer thickness  

                        smaple_rate : = 33
                            데이터 수집 속도 (Hz)

                        height :
                            총 적층 높이 
        
        Returns:    None
                    plot:

        
        Requirment: pip install pands
                    pip install matplotlib
        
        Example: 
                    >>>
    """
    mean_layer_temp, max_layer_temp, layer_list= temperature_per_layer(data, max(data["z"]), layer_thickness)

    fig = plt.figure(figsize=(16,9), tight_layout=True)
    fig.suptitle("Melt Pool Temperature Analysis")
    grid = plt.GridSpec(nrows=2, ncols=3, wspace=0.2, hspace=0.3)

    ################## ax1 : time series plot ###############
    ax1 = plt.subplot(grid[1, :2])
    series = ax1.plot(data.index/sample_rate, data["T"], color='tomato', linewidth=0.2, label='melt pool temperature')
    mean = ax1.plot([data.index[0]/sample_rate, data.index[-1]/sample_rate], [data["T"].mean(), data["T"].mean()]
                        , color='limegreen',linestyle="-.", linewidth=1.2,label="average" )
    ax1.set_title("Time series of melt pool temperature", fontsize=15)
    ax1.set_xlabel("time(s)")
    ax1.set_ylabel("temperature(℃)")
    ax1.set_ylim([min(data["T"])-100, max(data["T"])+100])
    ax1.legend()

    ############## ax2 : average temp of layer height plot ###############
    ax2 = plt.subplot(grid[1,2])
    error = np.asarray(max_layer_temp) - np.asarray(mean_layer_temp)
    bar_plot = ax2.barh(layer_list ,mean_layer_temp, height=0.2, color='gray', label="temperature")
    # , xerr=error, capsize=0.5, linewidth=0.07)
    ax2.grid(True, axis="x", linestyle="--")
    ax2.set_xlim([1000, 2400])
    ax2.set_title("Average Temperature per layer heigth", fontsize=15)
    ax2.set_xlabel("temperature(℃)")
    ax2.set_ylabel("height(mm)")

    ############## ax3 : histogram of melt pool temperature ###############
    ax3 = plt.subplot(grid[0,0])
    hist = ax3.hist(data['T'], bins=100, label='hist', color='gray', rwidth=0.6)
    ax3.grid(True, axis="y")
    ax3.set_title("histogram of melt pool temperature", fontsize=15)
    ax3.set_xlabel("temperature(℃)")
    ax3.set_ylabel("intensity")

    ############## ax4 : x-y plane temperature visualiztion ###############
    ax4 = plt.subplot(grid[0,1])
    layer = data.query("z == {}".format(height))
    layer = ax4.scatter(layer.x, layer.y, c=layer['T'], marker='s', cmap='jet', s=3, vmin=1000, vmax=2400)
    ax4.grid(True)
    ax4.set_ylim(0,100)
    ax4.set_xlim(0,100)
    ax4.set_xlabel("x position(mm)")
    ax4.set_ylabel("y position(mm)")
    ax4.set_title("x-y plane at hiegth {} mm".format(height), fontsize=15)
    fig.colorbar(layer)

    ############## ax5 : blank ###############
    import scipy as sp
    import scipy.stats
    ax5 = plt.subplot(grid[0,2])
    rv = sp.stats.norm(loc=data['T'].mean(), scale=data['T'].std())
    analysis = data.sort_values(by=['T'], axis=0)
    
    pdf = ax5.plot(analysis['T'], rv.pdf(analysis['T']), color='r')
    ax5.set_xlabel('temperature')
    ax5.set_ylabel('intensity')
    ax5.set_title('probability density function', fontsize=15)

    plt.show()

def xz_plane():
    load_file = "C:/Users/KAMIC/Desktop/VScode/repair_sleeve.csv"
    data = TDMs(load_file)
    xx, yy, zz = data_transform(data.x, data.y, data.z, data.c, data.a)
    data['x'] = xx
    data['y'] = yy
    data['z'] = zz
    data['z'] = data['z'].round(2)
    print(data.describe())
    print(data.z)

    t_data = data[data['z']==250.25]
    t_data['x_new'] = t_data['x'].round(1)
    t_data['y_new'] = t_data['y'].round(1) 

    t_data_gridded = t_data.groupby(['x_new', 'y_new'])['T'].mean().reset_index()
    from itertools import product
    x_array = np.arange(-100 ,100, 0.1)
    y_array = np.arange(-100, 100, 0.1)
    zero_df = pd.DataFrame(list(product(x_array, y_array)), columns=('x', 'y'))
    print(zero_df)
    # zero_df = zero_df.merge(t_data_gridded['x_new', 'y_new', 'T'], 
    #             left_on=('x', 'y'), 
    #             right_on=('x_new', 'y_new'), 
    #             how = "left")

    zero_df = pd.merge(zero_df, t_data_gridded, 
                left_on=('x', 'y'), 
                right_on=('x_new', 'y_new'), 
                how = "left")

    pivot = zero_df.pivot_table(values='T', index='x', columns='y')

    print(pivot)

    plt.imshow(pivot, cmap='jet')
    plt.show()



if __name__ =='__main__':
    ## user input ##
    # import time
    # load_file = "C:/Users/KAMIC/Desktop/VScode/repair_sleeve.csv"
    # data = TDMs(load_file)
    # xx, yy, zz = data_transform(data.x, data.y, data.z, data.c, data.a)

    # meltpool_temperature_spatial_3d_map(xx, yy, zz, data['T'])
    # print("Melt pool temerature analysis ver.1")
    # time.sleep(1)
    # print("1. Select your focas data!")
    # data = focas()
    # layer_thickness = float(input("2.layer thickness(mm) :"))
    # sample_rate = float(input("Sample rate(Hz) : "))
    # height = float(input("insert x axis height(mm) to drawing x-y plane : "))
    # print("wait a few second..........")
    print(visualization_data.__doc__)

    # pdf_plot(data)
    # xz_plane()

