import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

data = pd.read_table("20181211_100.log",names=["datetime","MAC","type","RSSI","UUID"])

subdata = data.loc[(data['MAC']=='12:3b:6a:1a:75:5a')]
print(subdata)

# x = data.datetime
# y = data.RSSI
# plt.plot(x, y)

subdata.plot()
plt.show()
