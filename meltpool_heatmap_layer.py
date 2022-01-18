import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
from melt_pool_3D_plot import Data_load

data = Data_load()


def meltpool_layer_ver1(data):
    resolution = 200
    step = 100 / resolution
    zero = np.zeros((100,100))
    zero_df = pd.DataFrame(zero)

    # layer_tickness = 0.25
    # layer_list = np.arange(0, max(data['z']), layer_tickness)


    for x in zero_df.index:
        for y in zero_df.columns:
            data.query('z == 8.5')
            t_data = data.query('{}< x <={} and {} < y <= {}'.format(x, x+step, y, y+step))

            if len(t_data) != 0:
                t = np.sum(t_data['T']) / len(t_data)
                zero_df[x][y] = t
            

    fig, ax = plt.subplots()

    im = ax.imshow(zero_df, cmap='jet',vmin=1000, vmax=1800)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Layer : {}'.format(8.5))
    ax.set_ylim(0, 99)
    plt.tight_layout()
    plt.colorbar(im)
    plt.show()

def x_y_2D_scatter(data):
    data=data.query('z==0.75')
    print(data)

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

if __name__ == '__main__':
    x_y_2D_scatter(data)