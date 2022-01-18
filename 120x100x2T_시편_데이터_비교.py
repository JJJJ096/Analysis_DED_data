import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chart_studio.plotly as py
from plotly.offline import plot
import plotly.express as px
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline(connected=True)

# data Load  
def Data_load(): 
    file_1 = "C:/Users/KAMIC/Desktop/새 폴더/20211015160654 Thin_wallt(t).txt"
    file_2 = "C:/Users/KAMIC/Desktop/새 폴더/20201117144901_수정.txt"
    
    data_1 = pd.read_csv(file_1, header=None, names=['date', 'num', 'x','y','z','c','a','vol','T'])
    data_2 = pd.read_csv(file_2, header=None, names=['date', 'num', 'x','y','z','c','a','vol','T'])
    
    data_1 = data_1[['x','y','z','c','a','T']]
    data_2 = data_2[['x','y','z','c','a','T']]
    data_1 = data_1.query('T >= 1200')
    data_1 = data_1.query('T <= 1700')
    data_2 = data_2.query('T >= 1100')
    data_2 = data_2.query('T <= 1500')
    
    avg_1 = np.average(data_1['T'])
    avg_2 = np.average(data_2['T'])

    print("20211016_meltpool temperature:{}".format(round(avg_1),3))
    print("20201117_meltpool temperature:{}".format(round(avg_2),3))

    return data_1, data_2


# 3축으로 변환 된 좌표, 온도데이터를 이용하여 3차원 Plot을 그린다.
def plot_3D(x,y,z,t):
    fig = plt.figure(figsize=(12,10))
    ax = plt.axes(projection='3d')
    
    sctt = ax.scatter3D(x, y, z, alpha=1, c=t, cmap='jet', s=10, vmin=min(t), vmax=max(t))
    plt.title("Temperature")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    #ax.grid(False)
    #ax.set_axis_off()

    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))
    fig.colorbar(sctt, ax=ax, shrink=0.7, aspect=10)
    ax.view_init(0, 90)                                        # 그래프 각도 view_init(z축, x-y축)
    plt.tight_layout()
    plt.show()

# plotly 3D Scatter plot
def plotly_plot(data, x,y,z,t):
    #data = [go.Scatter3d(x=data['x'], y=data['y'], z=data['z'], mode='markers', marker=dict(size=1, color=data['T'], colorscale='Jet', opacity=0.8))]
    #layout = go.Layout(autosize=False, width=1000, height=1000, )
    #fig = go.Figure(data = data, layout=layout)

    fig = px.scatter_3d(data, x='x', y='y', z='z', color='T', size_max=1, color_discrete_map='Jet', opacity=0.8)
    fig.show()
    #plot(fig, filename="plotly_test.html", auto_open=False)

def main():
    data_1, data_2 = Data_load()

    x_1, y_1, z_1, t = data_1['x'], data_1['y'], data_1['z'], data_1['T']
    x_2, y_2, z_2, t = data_2['x'], data_2['y'], data_2['z'], data_2['T']
    #fig = plt.figure(figsize=(16,9))
    #plt.plot(data['T'], 'r', linewidth=0.5)
    #plot_3D(x_1, y_1, z_1, t)
    plotly_plot(data_1, x_1, y_1, z_1, t)
    #plt.show()

if __name__ == '__main__':
    main()