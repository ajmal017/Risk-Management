#!/usr/bin/env python
# coding: utf-8

# In[97]:


#import csv
#from collections import defaultdict

#columns = defaultdict(list) # each value in each column is appended to a list

#with open('DISH.csv') as f:
    #reader = csv.DictReader(f) # read rows into a dictionary format
    #for row in reader: # read a row as {column1: value1, column2: value2,...}
        #for (k,v) in row.items(): # go over each column name and value 
            #columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k
#f=(columns['Adj Close'])

#print(f)

#import numpy as np
#my_array = np.array(f)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
from scipy.stats import norm

dish = web.get_data_yahoo("DISH",
                            start = "2019-03-27",
                            end = "2020-03-27")

print(dish.tail())

dish_daily_returns = dish['Adj Close'].pct_change()
pnl = dish_daily_returns*1000000
#print(dish_daily_returns)
#normal curve and histogram
#mean,std=norm.fit(dish_daily_returns)
#plt.hist(dish_daily_returns, bins=10, density=True)
#plt.hist(pnl, bins=10, density=True)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 10, 1)
n, bins, patches = ax1.hist(dish_daily_returns)
ax1.set_xlabel('Returns')
ax1.set_ylabel('Frequency')

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 10, 1)
n, bins, patches = ax2.hist(pnl)
ax2.set_xlabel('PnL')
ax2.set_ylabel('Frequency')
#xmin, xmax = plt.xlim()
#x = np.linspace(xmin, xmax, 100)
#y = norm.pdf(x, mean, std)
#plt.plot(x, y)
#plt.show()
array_length = len(dish_daily_returns)
print(array_length)
five = array_length*0.05
#print(five)
rounded = round(five)
#print (rounded)
descending = sorted(dish_daily_returns)
#print (descending)
VaR1 = (descending[13]*100)
#Historical 1yr 5% VaR for DISH
print ('The 5% percent VaR is: ', VaR1,'%')
#Parametric 1yr % VaR for DISH
mean = np.mean(dish_daily_returns)
sigma = np.std(dish_daily_returns)
parametric = (mean - 1.645*sigma)*100
print ('The 5% percent VaR is: ', parametric,'%')


# In[ ]:





# In[ ]:




