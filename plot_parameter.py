from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as fd

x = [0, 30, 30, 60, 60, 90]
y = [300, 300, 400, 400,300, 300]

plt.rcParams['font.family'] = 'Times New Roman' # 글꼴
plt.rc('font', size=20)                         # 기본 폰트 크기
plt.rc('axes', labelsize=30)                    # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=20)                   # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=20)                   # y축 눈금 폰트 크기
plt.rc('legend', fontsize=20)                   # 범례 폰트 크기
plt.rc('figure', titlesize=50)                  # figure title 폰트 크기

plt.plot(x, y, color='r', linewidth=3)
plt.xlabel("distance(mm)")
plt.ylabel("Laser Power(K)")
plt.tight_layout()
plt.ylim([200, 500])


plt.show()

