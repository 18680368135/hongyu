import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# df = pd.read_csv('50daysresult.csv')

x = np.linspace(0, 1, 100)
y1 = 0.5 * np.log(1-x)
y2 = -0.5 * np.log(x)
y3 = (x-1)**2

# print(title[0])
# print(df.iloc[:, 0].values)

plt.plot(x, y1, color='blue', label="minimax")
plt.plot(x, y2, color='red', label="heuristic")
plt.plot(x, y3, color='black', label="least-square")
plt.legend()

# plt.xlabel('D(G(^))')
# plt.ylabel('MG')
plt.savefig('./mutation.jpg')
plt.show()
