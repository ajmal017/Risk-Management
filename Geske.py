#!/usr/bin/env python
# coding: utf-8

# In[14]:


#Author: Cameron R. Shabahang

###Implement two-period Geske model

#Collect one year daily stock prices for 10 companies
#Compute daily returns, and then mean and standard deviation

import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import math
import scipy as sp
from sklearn.decomposition import PCA

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

#Filter function to obtain the dailyreturns
def filter(df):
    z = df[['Date','Adj Close']]
    close = z['Adj Close'].astype(float)
    dailyreturns = close.pct_change(1)
    z.insert(2,"Daily Return",dailyreturns)
    return z
df1, df2, df3, df4, df5, df6, df7, df8, df9, df10 = [filter(df) for df in (df1, df2, df3, df4, df5, df6, df7, df8, df9, df10)]
frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]
df_merge = pd.concat(frames, axis=1)
returns = df_merge['Daily Return']
returns = returns.drop(axis = 0, index = 0)
returns
means = returns.mean(axis=1)
print('The means are as follows: \n', means)
stdev = returns.std(axis=1)
print('The standard deviations are as follows: \n', stdev)


# In[26]:


#Collect (1) market cap, (2) current and (3) long-term liabilities of each firm
#Manual entry from Bloomberg, use Current values for Mkt cap to reflect COVID-19 shock
#JPM, GS, MS, C, CS, UBS, JEF, LAZ, BCS, EVR
mkt_cap = np.array([284089.6, 63488.2, 60518.4, 94763.5, 22303.8, 41543.6, 3521.7, 2650.7, 17812.7, 2312.0])
total_liab=np.array([2426049.0, 902703.0, 812732.0, 1757212.0, 743581.0, 917476.0, 39706.9, 4958.0, 1074569.0, 1472.4])
ltm_liab = np.array([316240.0, 209077.0, 196642.0, 251299.0, 152005.0, 100381.0, 8337.1, 2234.8 , 94525.0, 592.3])
current_liab = total_liab - ltm_liab


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




