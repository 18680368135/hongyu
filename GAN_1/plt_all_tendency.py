import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('399001.csv')
title = df.columns.values
x = list(range(1, df.shape[0]+1))
# print(title[0])
# print(df.iloc[:, 0].values)
values = df.iloc[:, -1].values
print(np.max(df.iloc[:, -1].values))



plt.plot(x, values, color='blue', label=title[-1])

plt.legend()

plt.xlabel('day')
plt.ylabel('value')
plt.savefig('./tendency.jpg')
plt.show()
