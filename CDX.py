#!/usr/bin/env python
# coding: utf-8

# In[94]:


#Author: Cameron R. Shabahang

#Use the BS model to calculate expected losses of tranches of a CDX which is $1.25 billion in size  EL (i.e. mean of the distribution) of the whole asset pool is $100 million.  Simply set S0=100, volatility to be 0.3, risk-free rate to be 5% (i.e. 0.05), time horizon to be 1 year (T = 1).  Evaluate the expected losses of equity tranche (0~3%), mezzanine tranches (3~7, 7~10), senior tranche (10%~30%) and super-senior tranche (30%~100%).  
#Equity tranche value is BS(K1) – BS(K0), mezzanine value is BS(K2) – BS(K1), rtc/


import numpy as np
import pandas as pd
import math
from scipy.stats import norm

#Set parameters
EL = 100 #Mean of the distribution of the asset pool
CDX = 1250 #Notional value of the CDX
S0 = 100 #Initial stock price in B-S model
vol = 0.3 #Volatility
rf = 0.05 #Risk-free rate
T = 1 #Time horizon

#Set tranches
equity = 0.020
mezz1 = 0.050
mezz2 = 0.080
senior = 0.200
super_senior = 0.650

tranches = [equity, mezz1, mezz2, senior, super_senior]

def totalK(tranches):
    total = 0
    debt = np.zeros(len(tranches))
    for i in range (len(tranches)):
        debt[i] = CDX*tranches[i]
        total = total+debt[i]
    return debt
total_debt = totalK(tranches)
print('The component debt values are as follows: ', total_debt)

def d2(S0, vol, T, tranches):
    d_2 = np.zeros(len(tranches))
    for i in range (len(tranches)):
        d_2[i] = (np.log(total_debt[i])-np.log(S0))+((rf-0.5*vol**2)*T)/(vol*np.sqrt(T))
    return d_2
d2_list = d2(S0, vol, T, tranches)
print('d2:', d2_list)

d1_list = d2_list+vol*np.sqrt(T)
print('d1:', d1_list)

d = norm(loc=0.0, scale=1.0);
Nd1 = d.cdf(d1_list)
print('Nd1:', Nd1)
Nd2 = d.cdf(d2_list)
print('Nd2:', Nd2) 
Pd = 100*d.cdf(-d2_list)
for i in range(len(tranches)):
    print('The probability of default of tranche ', i, ' is : ', Pd[i],'%')
    print('The expected loss per $100 of tranche ', i, ' is ($): ', Pd[i])

#Value of the tranche
def E_val(tranches):
    z = []
    for i in range(len(tranches)):    
        E = np.zeros(len(tranches))
        E[i] = CDX*Nd1[i]-np.exp(rf*T)*S0*Nd2[i]
        z.append(E[i])
        print('E(t)', i, ':', E[i])
    return z
E_list = E_val(tranches)
for i in range(len(tranches)):  
    value = np.zeros(len(tranches))
    if i < len(tranches)-1:
        value[i]= E_list[i+1]-E_list[i]
    elif i == len(tranches):
        value[i]= CDX - sum(value[0:i-1])
    print('The value of tranche', i,  'is:' , value[i])    


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




