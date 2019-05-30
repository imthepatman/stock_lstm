import json
import sys
import pandas as pd
from data_ops import *


window_size = 61
interval_min = -100
interval_max = None

stock_name = "Infineon"
data_columns=["Close","Volume"]
datasets = get_datasets(stock_name,data_columns,True)

data = [pd.DataFrame(ds).values[interval_min:interval_max] for ds in datasets]

data_vals = data[0]
print(np.shape(data_vals))

data_filtered = filter_data(data_vals,21,5)
print(np.shape(data_filtered))

plt.plot(data_vals[:,0])
plt.plot(data_filtered[:,0])
plt.show()

plt.plot(data_vals[:,1])
plt.plot(data_filtered[:,1])
plt.show()