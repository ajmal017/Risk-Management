#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Cameron R. Shabahang
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





# In[72]:


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
print('The 95% parametric portfolio VaR is:', pVar(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10))


# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:


#Portfolio 0.95 decay SD for VaR
decay_port_mean0 = np.zeros(len(portfolio_return))
decay = decay_port_mean0.astype(float)
decay_port_mean1 = decay.tolist() 
decay_port_mean1[0]= sum(portfolio_return)/(len(portfolio_return))
decay_port_sd = np.zeros(len(portfolio_return))
decay_port_mean = [[i] for i in decay_port_mean1]
print(decay_port_mean)
for i in range (len(portfolio_return)):
    decay_port_mean[i]=sum(decay_port_mean[i-1])/len(portfolio_return)
    decay_port_sd[i]=0.95*decay_port_sd[i-1]**2+(1-0.95)*(sum(portfolio_return[i-1])-decay_port_mean[i])**2
print(decay_port_mean)
print(decay_port_sd)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


Calculate 99% Historical PnL VaR for the portfolio
positionsize = 100000
sharesize_c = positionsize/float(rawdata.iloc[0,5])
sharesize_d = positionsize/float(rawdata2.iloc[0,5])
historicalportfolioPnLVaR = sorted(finaldata['Daily Px Change'])[VaRobservation]*sharesize_d + sorted(finaldata2['Daily Px Change'])[VaRobservation]*sharesize_c
print('The 95% historical portfolio PnL VaR is: $', historicalportfolioPnLVaR)

#Calculate 95% Parametric PnL VaR for the portfolio
#d_Mu = np.mean(finaldata['Daily Px Change'])
#d_Sigma = np.std(finaldata['Daily Px Change'])
#c_Mu = np.mean(finaldata2['Daily Px Change'])
#c_Sigma = np.std(finaldata2['Daily Px Change'])
#p_Mu = 0.5 * d_Mu + 0.5 * c_Mu
#covar = np.cov(finaldata['Daily Px Change'],finaldata2['Daily Px Change'])[0,1]
#p_Sigma = np.sqrt(((0.5*0.5)*(c_Sigma*c_Sigma))+2*(0.5*0.5)*covar*(c_Sigma*d_Sigma)+(0.5*0.5)*(d_Sigma*d_Sigma))
# = (p_Mu - 1.645 * p_Sigma) * (sharesize_c + sharesize_d)
#print('The 95% parametric portfolio PnL VaR is: $', parametricportfolioPnLVaR)


# In[ ]:



#Implement Factor Model VaR
#Run principle component analysis and retrieve the first two factors
#95% of the eigenvalues
#Locate the 1% position 

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[ ]:


#HW2 EVT, EWMA(GARCH), and MC for Parametric Portfolio VaR
#EVT in VaR
#Find values for u and nu
u = as.double(1.645*portRetSd)
ord1 = sort(portRet, decreasing = TRUE)
nu = 1
while(ord1[nu]>u):
  nu=nu+1
evt = ord1[1:nu]
n = length(closes[,1])
a = 0.05
print(u)
print(nu)

#Estimate xi and beta
def est(xi, beta):
  y=(1+(xi*(evt-u)/beta))^(-1/(xi-1))
  return(-(1/beta)*sum(log(y)))
estimate=mle(est,start=list(xi = 1.5, beta = 1.5))
xi=as.double(estimate@coef[1]) 
beta=as.double(estimate@coef[2]) 
print(xi)
print(beta)

#Solve for u*
ustar<-u+(beta/xi)*(((n/nu)*(a))^(-xi)-1)
print(ustar)

#Simulate 1 day distribution
pca=princomp(returns)
eigvals<-pca$sdev^2
explained<-rep(0, length(eigvals))
for(i in 1:length(eigvals))
  explained[i]=eigvals[i]/sum(eigvals)

#95% test
nfacs95=0
esum95=0
while(esum95<0.95):
  esum95=esum95+explained[nfacs95+1]
  nfacs95=nfacs95+1

#6 factors required for 95%
varcov<-cov(pca$loadings)
facmean<-mean(pca$loadings)

#2 Days of Prices with EWMA
#window = 60 or 3 months
lambda=0.97
n=2
set.seed(-1000)
errors=rnorm(1000, 0, 1)
rets=rep(0, n+1)
prices=rep(0, n+1)
sigsq=rep(0, n+1)
eps=rep(0, n+1)
rets[1]=mean(portRet) #set to simulated mean
prices[1]=1000000
sigsq[1]=sd(portRet)^2 #set to simulated vol
eps[1]=sqrt(sigsq)*errors[1]
for(i in range 2:(n+1)):
  sigsq[i]=lambda*sigsq[i-1]^2+(1-lambda)*rets[i-1]^2
  eps[i]=np.sqrt(sigsq[i])*errors[i]
  rets[i]=eps[i]+lambda*rets[i-1]+(1-lambda)*eps[i-1]
  prices[i]=(1+rets[i])*prices[i-1])


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




