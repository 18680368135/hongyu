import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('50daysresult.csv')
prediction = df.iloc[:, 1]
real = df.iloc[:, 0]
plt.plot(prediction, color='r', label='prediction')
plt.plot(real, color='b', label='real')
plt.legend()
plt.show()

real = np.array(real)
prediction = np.array(prediction)

MSE = np.mean(np.square(real - prediction))
MAPE = np.mean(np.abs((real - prediction) / real)) * 100
MAE = np.mean(np.abs(real - prediction))

print('MSE:%.4f' % MSE)
print('MAPE:%.4f' % MAPE)
print('MAE:%.4f' % MAE)