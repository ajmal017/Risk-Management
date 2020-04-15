#!/usr/bin/env python
# coding: utf-8

# In[10]:


from scipy.stats import norm
import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import math
import scipy as sp
import datetime as dt

##Take a CMT series from FRED to estimate the parameters of the HL model

#Read and clean organize data
cmt = pd.read_csv("DGS3MO.csv",na_values = '.' , keep_default_na = True)
cmt = cmt[cmt['DGS3MO'].notna()]
#print(cmt.tail(10))
a,b = 0,0
cmt['DATE'] = pd.to_datetime(cmt['DATE'])
cmt['DATE']=cmt['DATE'].map(dt.datetime.toordinal)
cmt_dates = cmt['DATE']
cmt_rates = cmt['DGS3MO']

cmt_diff = cmt_rates.diff()
print(cmt_dates)
print(cmt_rates)
print (cmt_diff)
cmt_prices = cmt_rates +100
print(cmt_prices)
#Function to determine zero coupon price series
def zcb_price(cmt_dates, a,b):
    #Define tau and indicator variable
    tau = 0.25
    for i in cmt_diff:
        if cmt_diff[i]>0:
            ind = 1
        elif cmt_diff[i]<0:
            ind = 0
        else: 
            ind = 1
    # mean and standard deviation
    mu, sigma = 0, 1 
    u = np.random.normal(mu, sigma, 1)
    delta = np.exp(b/tau)
    p = (np.exp(-a)-delta**tau)/(1-delta**tau)
    np.log(cmt_diff+100) = np.log((p+(1-p)*delta**tau))+tau*np.log(delta)*(ind)+u 

#Fit the curve to estimate parameters
popt, pcov = sp.optimize.curve_fit(zcb_price, cmt_dates, cmt_prices)
print('The parameters a1 through a4 are as follows: ', popt)
print('\nThe variance-covariance matrix is as follows: ', pcov)





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




