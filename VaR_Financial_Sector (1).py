#!/usr/bin/env python
# coding: utf-8

# In[93]:


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
    return z

df1, df2, df3, df4, df5, df6, df7, df8, df9, df10 = [filter(df) for df in (df1, df2, df3, df4, df5, df6, df7, df8, df9, df10)]

#Establish list of dataframes
#A = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
#Loop through the dataframes to apply the filter function
#for i in range(0,len(A)): 
    #count = 0
    #A[i] = filter(A[i])
    #count = count + i
    #print(count)
#print(A)

#Compute daily returns (PnLs), means, and standard deviation
#def port(A):
    #for i in range(0,len(A)):    
        #port = A[i]+A[i-1]
        #portf = port['Weighted']
        #return portf

#Call the portfolio function and display    
#x = port(A)
#print(x)

#agg =  A[1]+A[2]+A[3]+A[4]+A[5]+A[6]+A[7]+A[8]+A[9]
#print (agg['Weighted


########Reference code for 2 stock portfolio

#Calculate 2 stock portfolio daily returns
#portfolio_return = 0.5 * finaldata['Daily Return'] + 0.5 * finaldata2['Daily Return']

#Calculate 95% historical VaR for the portfolio
#portfolioVaRobservation = round(0.05 * len(portfolio_return))
#portfoliohistoricalVaR = sorted(portfolio_return)[portfolioVaRobservation]*100
#print('The 95% historical VaR is:',portfoliohistoricalVaR,'%')

#Calculate 95% parametric VaR for the portfolio
#d_Mu = np.mean(finaldata['Daily Return'])
#d_Sigma = np.std(finaldata['Daily Return'])
#c_Mu = np.mean(finaldata2['Daily Return'])
#c_Sigma = np.std(finaldata2['Daily Return'])
#p_Mu = 0.5 * d_Mu + 0.5 * c_Mu
#covar = np.cov(finaldata['Daily Return'],finaldata2['Daily Return'])[0,1]
#p_Sigma = np.sqrt(((0.5*0.5)*(c_Sigma*c_Sigma))+2*(0.5*0.5)*covar*(c_Sigma*d_Sigma)+(0.5*0.5)*(d_Sigma*d_Sigma))
#portfolioParametricVaR = (p_Mu - 1.645 * p_Sigma) * 100
#print('The 95% parametric portfolio VaR is:', portfolioParametricVaR, '%')

#Calculate 95% Historical PnL VaR for the portfolio
#positionsize = 500000
#sharesize_c = positionsize/float(rawdata.iloc[0,5])
#sharesize_d = positionsize/float(rawdata2.iloc[0,5])
#historicalportfolioPnLVaR = sorted(finaldata['Daily Px Change'])[VaRobservation]*sharesize_d + sorted(finaldata2['Daily Px Change'])[VaRobservation]*sharesize_c
#print('The 95% historical portfolio PnL VaR is: $', historicalportfolioPnLVaR)

#Calculate 95% Historical Price Change VaR for the portfolio
#historicalportfolioPriceChangeVaR = sorted(finaldata['Daily Px Change'])[VaRobservation] + sorted(finaldata2['Daily Px Change'])[VaRobservation]
#print('The 95% historical portfolio Price Change VaR is: $', historicalportfolioPriceChangeVaR)

#Calculate 95% Parametric Price Change VaR for the portfolio
#d_Mu = np.mean(finaldata['Daily Px Change'])
#d_Sigma = np.std(finaldata['Daily Px Change'])
#c_Mu = np.mean(finaldata2['Daily Px Change'])
#c_Sigma = np.std(finaldata2['Daily Px Change'])
#p_Mu = 0.5 * d_Mu + 0.5 * c_Mu
#covar = np.cov(finaldata['Daily Px Change'],finaldata2['Daily Px Change'])[0,1]
#p_Sigma = np.sqrt(((0.5*0.5)*(c_Sigma*c_Sigma))+2*(0.5*0.5)*covar*(c_Sigma*d_Sigma)+(0.5*0.5)*(d_Sigma*d_Sigma))
#parametricportfolioPxChangeVaR = (p_Mu - 1.645 * p_Sigma)
#print('The 95% parametric portfolio Price Change VaR is: $', parametricportfolioPxChangeVaR)

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




