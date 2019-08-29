import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('realFakeloss.csv')
loss = df.iloc[:, 0]
v_loss = df.iloc[:, 1]
plt.plot(loss, color='r', label='loss1')
plt.plot(v_loss, color='b', label='v_loss')
plt.legend()
plt.show()
