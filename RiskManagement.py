#!/usr/bin/env python
# coding: utf-8

# In[60]:


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

dish = web.get_data_yahoo("DISH",
                            start = "2019-03-27",
                            end = "2020-03-27")

print(dish.head())

dish_daily_returns = dish['Adj Close'].pct_change()
#print(dish_daily_returns)
plt.hist(dish_daily_returns, bins=10)
array_length = len(dish_daily_returns)
#print(array_length)
five = array_length*0.05
#print(five)
rounded = round(five)
#print (rounded)
descending = sorted(dish_daily_returns)
#print (descending)
VaR1 = (descending[13])
print ('The 5% percent VaR is: ', VaR1)


# In[ ]:





# In[ ]:




