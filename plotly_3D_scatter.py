import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chart_studio.plotly as py
from plotly.offline import plot
import plotly.express as px
import plotly.graph_objs as go
import cufflinks as cf

min_temperature, max_temperature = 1100 , 1800

file = "C:/Users/KAMIC/Desktop/새 폴더/20211015160654 Thin_wallt(t).txt"
data = pd.read_csv(file, header=None, names=['date', 'num', 'x','y','z','c','a','vol','T'])
data = data[['x','y','z','c','a','T']]

data = data.query('{} <= T <= {}'.format(min_temperature, max_temperature))

print(np.shape(data))
print(data)
x = np.asarray(data['x'])
y = np.asarray(data['y'])
z = np.asarray(data['z'])
t = np.asarray(data['T'])
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1, color=t, colorscale='Jet'))])

#layout = go.Layout(autosize=False, width=1000, height=1000, )
#fig = go.Figure(data = data, layout=layout)

#fig = px.scatter_3d(data, x='x', y='y', z='z', color='T', size_max=1, color_discrete_map='Jet', opacity=0.8)
fig.show()
#plot(fig, filename="plotly_test.html", auto_open=False)