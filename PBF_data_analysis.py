import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


def PBF_data():
    file_name = "C:/Users/KAMIC/Downloads/CoolTerm Capture 2022-01-27 19-06-30.txt"

    data = pd.read_csv(file_name, header=None,names=["x", "y", "PD1", "PD2"])
    
    data['time'] = data['x'].str[:19]
    data["x"] = data["x"].str[21:]
    data['x'] = data['x'].replace('X','')
    data['x'] = data['x'].replace('RRORX','')
    data['y'] = data['y'].replace('Y','')
    # data["x"] = data["x"].str[21:]
    # data["y"] = data["y"].str[2:]
    
    data = data.drop([0,1])

    # data['x'] = pd.to_numeric(data['x'])
    # data['y'] = pd.to_numeric(data['y'])
    # print(type(data[0]))
    # data = data.astype(float)
    # data["x"] = data["x"].astype(float)
    # data["y"] = data["y"].astype(float)
    print(data)
    # print(data.describe())
    return data

def plot(data):
    fig = plt.figure()
    ax1 = plt.subplot(211)
    PD1 = ax1.plot(data.index, data["PD1"])
    ax1.set_ylim([-1, 3])
    ax1.set_xlabel("index")
    

    # ax2 = plt.subplot(212)
    # PD2 = ax2.scatter(data.x, data.y, c=data["PD2"])
    # ax2.set_xlim(0, 3)
    # ax2.set_ylim()

    plt.show()
if __name__ == "__main__":
    plot(PBF_data())
    # PBF_data()
