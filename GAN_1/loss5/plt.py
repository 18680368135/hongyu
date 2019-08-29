import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('result.csv')
title = df.columns.values
x = list(range(1, df.shape[0]+1))
# print(title[0])
# print(df.iloc[:, 0].values)
real_values = df.iloc[:, 1].values
predict_values = df.iloc[:, 0].values
MSE = np.mean(np.square(real_values - predict_values))
MAPE = np.mean(np.abs((real_values - predict_values) / real_values)) * 100
MAE = np.mean(np.abs(real_values - predict_values))

print('MSE:%.4f' % MSE)
print('MAPE:%.4f' % MAPE)
print('MAE:%.4f' % MAE)
plt.plot(x, real_values, color='blue', label=title[1])
plt.plot(x, predict_values, color='red', label=title[0])
plt.legend()

plt.xlabel('day')
plt.ylabel('value')
plt.savefig('./300daysresult.jpg')
plt.show()
