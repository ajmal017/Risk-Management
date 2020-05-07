#!/usr/bin/env python
# coding: utf-8

# In[15]:


#Author: Cameron R. Shabahang
#Risk Management

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

#Form a portfolio with chosen weights and shares
def filter(df):
    z = df[['Date','Adj Close']]
    close = z['Adj Close'].astype(float)
    dailyreturns = close.pct_change(1)
    z.insert(2,"Daily Return",dailyreturns)
    z['Daily Px Change'] = close - close.shift(1)
    z = z.iloc[1:]
    z['Weighted'] = z['Daily Return'].apply(lambda x: x*0.1)
    return z

df1, df2, df3, df4, df5, df6, df7, df8, df9, df10 = [filter(df) for df in (df1, df2, df3, df4, df5, df6, df7, df8, df9, df10)]
portfolio_return = df1['Weighted']+df2['Weighted']+df3['Weighted']+df4['Weighted']+df5['Weighted']+ df6['Weighted']+df7['Weighted']+df8['Weighted']+df9['Weighted']+df10['Weighted']
print(portfolio_return)

#Calculate 99% historical VaR for the portfolio
portfolioVaRobservation = round(0.01 * len(portfolio_return))
portfoliohistoricalVaR = sorted(portfolio_return)[portfolioVaRobservation]*100
print('The 99% historical VaR is:',portfoliohistoricalVaR,'%')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[77]:


#Calculate 99% parametric VaR for the portfolio
def pVar(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10):
    weights = np.repeat(0.10, 252)
    #df_wMu = 0.1*((np.mean(df1['Daily Return'])+np.mean(df2['Daily Return'])+np.mean(df3['Daily Return'])+np.mean(df4['Daily Return'])+ #Needs to take df1-df10
                  #np.mean(df4['Daily Return'])+np.mean(df5['Daily Return'])+np.mean(df6['Daily Return'])+np.mean(df7['Daily Return'])+
                  #np.mean(df8['Daily Return'])+np.mean(df9['Daily Return'])+np.mean(df10['Daily Return']))
    frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]
    df_merge = pd.concat(frames, axis=1)
    z=df_merge['Daily Return']
    print(z)
    covar = np.cov(df_merge['Daily Return'])
    pvar = np.dot(weights.T,np.dot(covar,weights)) #Takes covariance of df1-df10
    portfolioParametricVaR = 2.326*np.sqrt(pvar)
    return portfolioParametricVaR
print('The 99% parametric portfolio VaR is:', pVar(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10)*100, '%')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:


#Portfolio 0.95 decay SD for VaR

#Portfolio 0.95 decay SD for VaR
decay_mean = np.zeros(252)
decay_mean_shift = np.zeros(252)
decay_var = np.zeros(252)
for i in range (len(portfolio_return)):
    if i == 0:
        decay_var[i] = 0
    elif i == 1:
        rollingreturn = portfolio_return[0:i + 1]
        rollingreturn_shift = portfolio_return[0:i]
        decay_mean[i] = np.mean(rollingreturn)
        decay_mean_shift[i] = np.mean(rollingreturn_shift)
        decay_var[i] = 0.95 * decay_var[i - 1]+(1 - 0.95) * (rollingreturn_shift[i] - decay_mean[i]) ** 2
    else:
        rollingreturn = portfolio_return[0:i + 1]
        rollingreturn_shift = portfolio_return[0:i]
        decay_mean[i] = np.mean(rollingreturn)
        decay_mean_shift[i] = np.mean(rollingreturn_shift)
        decay_var[i] = 0.95 * decay_var[i - 1]+(1 - 0.95) * (rollingreturn_shift[i] - decay_mean[i]) ** 2
print('The 0.95 decay VaR is: ', np.sqrt(decay_var[251])*100, '%')


# In[ ]:





# In[69]:


#Calculate 99% historical VaR for the portfolio
portfolioVaRobservation = round(0.01 * len(portfolio_return))
portfoliohistoricalVaR = sorted(portfolio_return)[portfolioVaRobservation]*100
print('The 99% historical VaR is:',portfoliohistoricalVaR,'%')


# In[ ]:





# In[68]:


#Implement Factor Model VaR
#Run principle component analysis and retrieve the first two factors
#95% of the eigenvalues
#Locate the 1% position 

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

X = portfolio_return
X, y = make_classification(n_samples=1000)
n_samples = X.shape[0]

pca = PCA()
X_transformed = pca.fit_transform(X)

# We center the data and compute the sample covariance matrix.
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
eigenvalues = pca.explained_variance_
count = 0
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
    print('\n', np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    print(eigenvalue)
    count = count+1
print('\nThe number of eigenvalues is: ', count)


# In[ ]:





# In[ ]:





# In[19]:


#HW2 EVT, EWMA(GARCH), and MC for Parametric Portfolio VaR
#EVT in VaR
#Find values for u and nu
port_SD = pVar(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10)/2.326
u = 1.645*port_SD
order = portfolio_return.sort_values(ascending=False)
order = order.reset_index(drop=True)
print(order)   
i=0
while(order[i]>u):
    i=i+1
evt = order[0:i]
print('EVT list is:\n', evt)
n = len(portfolio_return)
a = 0.05
nu = i
print('The value of u is: ', u)
print('The value of nu is: ', nu)


# In[ ]:





# In[27]:


from scipy.optimize import minimize
#Equation 10.9, p. 156
#Estimate xi and beta in EVT
xi = 0
beta = 0
def lik(x, n, nu, evt, u):
    y=(n/nu)*(1+(x[1]*(evt-u)/x[0]))**(-1/(x[1]))
    return(-(1/x[0])*sum(np.log(y)))
lik_model = minimize(lik,[1.5,1.5], args =(n, nu, evt, u),  method='SLSQP')
print(lik_model)
beta = lik_model.x[0]
xi = lik_model.x[1]
#Solve for u*
ustar = u+(beta/xi)*(((n/nu)*(a))**(-xi)-1)
print('ustar is as follows: ', ustar)


# In[75]:



#Simulate 1 day distribution
X = portfolio_return
X, y = make_classification(n_samples=1000)
n_samples = X.shape[0]

pca = PCA()
X_transformed = pca.fit_transform(X)

# We center the data and compute the sample covariance matrix.
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
eigenvalues = pca.explained_variance_
count = 0
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
    print('\n', np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    print(eigenvalue)
    count = count+1
print('\nThe number of eigenvalues is: ', count)

#Retrive until 95% of the eigenvalues
n95=0
e95=0
while(e95<0.95):
    e95=e95+eigenvalues[n95+1]
    n95=n95+1
print(e95)
print('The count is: ', n95)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[71]:


import random 
#Simulate two days of prices using EWMA (GARCH)
#Equation 10.3 on page 153
lam=0.97
n=2
errors=np.random.normal(0, 1, [3,1])
rets=np.zeros((n+1))
prices=np.zeros((n+1))
sig_sq=np.zeros((n+1))
eps=np.zeros(n+1)

rets[0]=np.mean(portfolio_return) 
prices[0]=1000000
sig_sq[0]=np.std(portfolio_return)**2 
eps[0]=np.sqrt(sig_sq[0])*errors[0]


for i in range (1,n+1):
    sig_sq[i]=lam*sig_sq[i-1]**2+(1-lam)*rets[i-1]**2
    eps[i]=np.sqrt(sig_sq[i])*errors[i]
    rets[i]=eps[i]+lam*rets[i-1]+(1-lam)*eps[i-1]
    prices[i]=(1+rets[i])*prices[i-1]
print('The two days of prices are as follows: ' , prices)


# In[ ]:




