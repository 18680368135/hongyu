import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

df = pd.read_csv('399001.csv')
title = df.columns.values
x = list(range(1, df.shape[0]+1))
# print(title[0])
# print(df.iloc[:, 0].values)
values = df.iloc[:, -1].values

plt.figure()
p1 = plt.subplot(211)
p2 = plt.subplot(234)
p3 = plt.subplot(235)

p1.plot(x, values, color='blue', label=title[-1])
p2.plot(x, values, color='red', label=title[-1])
p2.xaxis.set_major_locator(MultipleLocator(10))
p2.xaxis.set_minor_locator(MultipleLocator(1))

p3.plot(x, values, color='red', label=title[-1])
p3.xaxis.set_major_locator(MultipleLocator(1))
p3.xaxis.set_minor_locator(MultipleLocator(1))


p1.axis([-50, 4500, np.min(df.iloc[:, -1].values)-100, np.max(df.iloc[:, -1].values)+100])
p1.set_ylabel("value", fontsize=14)
p1.set_xlabel("day", fontsize=14)

p1.grid(True)
p1.legend()

p2.axis([df.shape[0]-50, df.shape[0], np.min(df.iloc[-50:-1, -1].values)-100, np.max(df.iloc[-50:-1, -1].values)+100])
p2.set_ylabel("value", fontsize=14)
p2.set_xlabel("day", fontsize=14)

p2.grid(True, which="minor")
p2.legend()

p3.axis([df.shape[0]-10, df.shape[0], np.min(df.iloc[-10:-1, -1].values)-100, np.max(df.iloc[-10:-1, -1].values)+100])
p3.set_ylabel("value", fontsize=14)
p3.set_xlabel("day", fontsize=14)

p3.grid(True, which="minor")
p3.legend()

# plot the box
tx0 = df.shape[0]-50
tx1 = df.shape[0]
ty0 = int(np.min(df.iloc[-50:-1, -1].values)) - 50
ty1 = int(np.max(df.iloc[-50:-1, -1].values)) + 50

sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
p1.plot(sx,sy,"purple")

# plot patch lines
xy=(df.shape[0]-57, ty1-1)
xy2 = (4497, ty1-1)
con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
        axesA=p2, axesB=p1)
p2.add_artist(con)

xy = (df.shape[0]-57, ty0+1)
xy2 = (4497, ty0+1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=p2,axesB=p1)
p2.add_artist(con)


# plt.savefig('./tendency.jpg')
plt.show()
