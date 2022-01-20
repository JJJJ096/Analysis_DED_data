import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from data_processing import focas, TDMs, save_csv_to_tdms, data_transform, temperature_per_layer

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
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
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


if __name__ =='__main__':
    data = focas()
    time_series(data, 33)