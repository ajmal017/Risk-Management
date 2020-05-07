#!/usr/bin/env python
# coding: utf-8

# In[72]:


#Author: Cameron R. Shabahang

###Implement two-period Geske model

#Collect one year daily stock prices for 10 companies
#Compute daily returns, and then mean and standard deviation

import pandas as pd
import numpy as np
import scipy.stats as stat
import math
import scipy as sp
from scipy.stats import norm
from scipy.stats import mvn

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
means = returns.mean(axis=0)
means = means.reset_index(drop = True)
print('The means are as follows: \n', means)
stdev = returns.std(axis=0)
stdev = stdev.reset_index(drop=True)
print('The standard deviations are as follows: \n', stdev)


# In[ ]:





# In[73]:


#Collect (1) market cap, (2) current and (3) long-term liabilities of each firm
#Manual entry from Bloomberg, use Current values for Mkt cap to reflect COVID-19 shock
#JPM, GS, MS, C, CS, UBS, JEF, LAZ, BCS, EVR
mkt_cap = np.array([284089.6, 63488.2, 60518.4, 94763.5, 22303.8, 41543.6, 3521.7, 2650.7, 17812.7, 2312.0])
total=np.array([2426049.0, 902703.0, 812732.0, 1757212.0, 743581.0, 917476.0, 39706.9, 4958.0, 1074569.0, 1472.4])
ltm = np.array([316240.0, 209077.0, 196642.0, 251299.0, 152005.0, 100381.0, 8337.1, 2234.8 , 94525.0, 592.3])
current = total - ltm
print(current)


# In[ ]:





# In[ ]:





# In[ ]:





# In[74]:


mkt_cap


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[85]:


#Geske model implementation

#We use periods 1 and 2
r=0.05
t=0 
t1=1
t2=2
rho = np.sqrt((t1-t)/(t2-t))

hpos1=np.zeros(len(mkt_cap))
hpos2=np.zeros(len(mkt_cap))
hneg1=np.zeros(len(mkt_cap))
hneg2=np.zeros(len(mkt_cap))
mpos=np.zeros(len(mkt_cap))
mneg=np.zeros(len(mkt_cap))
assetval=np.zeros(len(mkt_cap))
volatility=np.zeros(len(mkt_cap))

assetval = []
for i in range(len(mkt_cap)):
    hpos1[i]=math.log(mkt_cap[i])-math.log(current[i])+(r+stdev[i]**2/2)*(t1-t)/(stdev[i]*np.sqrt(t1-t))
    hpos2[i]=math.log(mkt_cap[i])-math.log(ltm[i])+(r+stdev[i]**2/2)*(t2-t)/(stdev[i]*np.sqrt(t2-t))
    hneg1[i]=math.log(mkt_cap[i])-math.log(current[i])+(r-stdev[i]**2/2)*(t1-t)/(stdev[i]*np.sqrt(t1-t))
    hneg2[i]=math.log(mkt_cap[i])-math.log(ltm[i])+(r-stdev[i]**2/2)*(t2-t)/(stdev[i]*np.sqrt(t2-t))
    
    low = np.array([-100, -100])
    upp = np.array([hpos1[i], hpos2[i]])
    mu = np.array([0, 0])
    S = np.array([[hpos1[i],rho],[rho,hpos1[i]]])
    mpos,i = mvn.mvnun(low,upp,mu,S)
    print ('Positive bivariate standard normal probability: ', mpos)
    
    low1 = np.array([-100, -100])
    upp1 = np.array([hneg1[i], hneg2[i]])
    mu1 = np.array([0, 0])
    S1 = np.array([[hneg1[i],rho],[rho,hneg1[i]]])
    mneg,i = mvn.mvnun(low1,upp1,mu1,S1)
    print ('Negative bivariate standard normal probability: ', mneg)

    #Equation 14.4 for asset value and volatility
    val = np.zeros(len(mkt_cap))
    val[i] = mkt_cap[i]*mpos-np.exp(-r*(t2-t))*ltm[i]*mneg-np.exp(-r*(t2-t))*current[i]*norm.pdf(hneg1[i])
    assetval.append(val[i])
    #volatility[i]=stdev[i]*mpos*(mkt_cap[i]/assetval)
print('\nThe asset value E(t) is: ', assetval)
print('The volatility (sigma) is: ', volatility, '\n')
print(hpos1)
print(hpos2)
print(hneg1)
print(hneg2)
print(rho)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[82]:


#Compare with book value
bookval=np.zeros(len(mkt_cap))
for i in range(len(bookval)):
    bookval[i]=mkt_cap[i]-(current[i]+ltm[i])
print(bookval)


# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


#Calculate default probability
default=np.zeros(len(mkt_cap))
for i in range(1,len(default)):
    d2=math.log(assetval[i])-math.log(current[i]+ltm[i])/volatility[i]
    default[i]=1-norm.pdf(d2)


# In[ ]:





# In[ ]:





# In[ ]:




