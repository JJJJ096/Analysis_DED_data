import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


case_1 = pd.read_csv("C:/Users/KAMIC/Desktop/VScode/Analysis_DED_data/machine_noise/case1.txt", header=None, names=['0','time(s)','data', 'dt', 'PD', 'laser'], encoding='CP949', sep=' ')
case_2 = pd.read_csv("C:/Users/KAMIC/Desktop/VScode/Analysis_DED_data/machine_noise/case2.txt", header=None, names=['0','time(s)','data', 'dt', 'PD', 'laser'], encoding='CP949', sep=' ')
case_3 = pd.read_csv("C:/Users/KAMIC/Desktop/VScode/Analysis_DED_data/machine_noise/case3.txt", header=None, names=['0','time(s)','data', 'dt', 'PD', 'laser'], encoding='CP949', sep=' ')

case_1 = case_1[['time(s)','PD', 'laser']]
case_2 = case_2[['time(s)','PD', 'laser']]
case_3 = case_3[['time(s)','PD', 'laser']]

fig = plt.figure()


plt.plot(case_1.index, case_1['PD'],  'g', label = 'PD', linewidth=0.3)
plt.plot(case_1['laser'], 'r', label = 'laser', linewidth=0.3)
plt.xlabel("time(s)")
plt.ylabel("Voltage(V)")

plt.tight_layout()
plt.legend()
plt.show()