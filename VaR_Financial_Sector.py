#!/usr/bin/env python
# coding: utf-8

# In[32]:


#Cameron R. Shabahang
#Risk Management

import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import math
import scipy as sp

#Choose 10 stocks and 1 year historical daily prices
df1 = pd.read_csv('JPM.csv', dtype = object)
df2 = pd.read_csv('GS.csv', dtype = object)
df3 = pd.read_csv('MS.csv', dtype = object)
df4 = pd.read_csv('C.csv', dtype = object)
df5 = pd.read_csv('CS.csv', dtype = object)
df6 = pd.read_csv('UBS.csv', dtype = object)
df7 = pd.read_csv('JEF.csv', dtype = object)
df8 = pd.read_csv('LAZ.csv', dtype = object)
df9 = pd.read_csv('BCS.csv', dtype = object)
df10 = pd.read_csv('EVR.csv', dtype = object)

#Form a portfolio with chosen weights and shares
def filter(df):
    z = df[['Date','Adj Close']]
    close = z['Adj Close'].astype(float)
    dailyreturns = close.pct_change(1)
    z.insert(2,"Daily Return",dailyreturns)
    z['Daily Px Change'] = close - close.shift(1)
    z = z.iloc[1:]
    z['Weighted'] = z['Daily Return'].apply(lambda x: x*0.1)
    #return print((z).head(5))

df1, df2, df3, df4, df5, df6, df7, df8, df9, df10 = [filter(df) for df in (df1, df2, df3, df4, df5, df6, df7, df8, df9, df10)]

print(df1, df2, df3, df4, df5, df6, df7, df8, df9, df10)


#for i in range(0,len(A)):
    #A[i] = your_func(A[i])

#Compute daily returns (PnLs), means, and standard deviation

#portfolio = df1['Weighted'] + df2['Weighted']
#portfolio 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




