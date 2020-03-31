#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import math
import scipy as sp

#Read DISH prices from a csv file and drop into a dataframe
rawdata = pd.read_csv('DISH (2).csv', dtype = object)

#Inspect the data in the dataframe
rawdata.head()
rawdata.shape
rawdata.columns

#Clean up the data
finaldata = rawdata[['Date','Adj Close']]

#Inspect final data
finaldata.head()
finaldata.shape
finaldata.columns

#Add a new column to show daily return
AdjCloseFloat = finaldata['Adj Close'].astype(float)
dailyreturns = AdjCloseFloat.pct_change(1)
#print(dailyreturns)

finaldata.insert(2,"Daily Return",dailyreturns)

#Add a new column to show daily price change
finaldata['Daily Px Change'] = AdjCloseFloat - AdjCloseFloat.shift(1)

#Drop first row in daily % return
finaldata = finaldata.iloc[1:]
finaldata.head()
finaldata.shape

#Plot a histogram of daily returns
#plt.hist(finaldata['Daily Return'],bins = 10)
#plt.xlabel('Daily % Return')
#plt.ylabel('Frequency')

#Calculate 95% historical VaR for DISH
VaRobservation = round(0.05 * len(finaldata['Daily Return']))
historicalVaR = sorted(finaldata['Daily Return'])[VaRobservation]*100
print('The 95% historical Var is:',historicalVaR,'%')

#Calculate 95% parametric VaR for DISH
mu = np.mean(finaldata['Daily Return'])
sigma = np.std(finaldata['Daily Return'])
parametricVaR = (mu - 1.645 * sigma) * 100
print('The 95% parametric VaR is:', parametricVaR,'%')

#Plot a normal distribution chart of returns
#x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
#y = stat.norm.pdf(x, mu, sigma)
#plt.plot(x, y)
#plt.show()

#Draw a daily price change PnL histogram
#plt.hist(finaldata['Daily Px Change'],bins = 10)
#plt.xlabel('Daily Price Change')
#plt.ylabel('Frequency')

#Draw a position PnL histogram
positionsize = 1000000
sharesize = positionsize/float(rawdata.iloc[0,5])
#plt.hist(finaldata['Daily Px Change'] * sharesize,bins = 10)
#plt.xlabel('Daily PnL')
#plt.ylabel('Frequency')

#Historical and Parametric 95% VaR PnL
historicalPxChangeVaR = sorted(finaldata['Daily Px Change'])[VaRobservation]*sharesize
muPxChange = np.mean(finaldata['Daily Px Change'])
sigmaPxChange = np.std(finaldata['Daily Px Change'])
parametricPxChangeVaR = (muPxChange - 1.645 * sigmaPxChange) *sharesize
print('The 95% historical VaR PnL is: $', historicalPxChangeVaR)
print('The 95% parametric VaR PnL is: $', parametricPxChangeVaR)

#Normal PDF Price Change
#x = np.linspace(muPxChange - 3 * sigmaPxChange, muPxChange + 3 * sigmaPxChange, 100)
#y = stat.norm.pdf(x, muPxChange, sigmaPxChange)
#plt.plot(x, y)
#plt.show()

#Normal PDF PnL
#x = np.linspace((muPxChange - 3 * sigmaPxChange)*sharesize, (muPxChange + 3 * sigmaPxChange)*sharesize, 100)
#y = stat.norm.pdf(x, muPxChange * sharesize, sigmaPxChange*sharesize)
#plt.plot(x, y)
#plt.show()

#Create a plot with subplots for 1 stock example
#Plt1: DISH Daily % Return Histogram
fig, ((plt1, plt2), (plt3,plt4), (plt5, plt6)) =  plt.subplots(3,2)
fig.suptitle ('DISH Daily PnL Analysis', fontsize = 16)
plt1.hist(finaldata['Daily Return'],bins = 10)
plt1.set(xlabel = 'Daily % Return', ylabel = 'Frequency')
plt1.set_title('DISH Daily % Return')
#Plt2: DISH Daily % Return Normal PDF
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
y = stat.norm.pdf(x, mu, sigma)
plt2.set(xlabel = 'Daily % Return', ylabel = 'Probability Density')
plt2.set_title('DISH Daily % Return Normal PDF')
plt2.plot(x, y)
#Plt3: DISH Daily Price Change Histogram
plt3.hist(finaldata['Daily Px Change'],bins = 10)
plt3.set(xlabel = 'Daily Price Change', ylabel = 'Frequency' )
plt3.set_title('DISH Daily Price Change Histogram')
#Plt4: DISH Daily PnL Histogram
positionsize = 1000000
sharesize = positionsize/float(rawdata.iloc[0,5])
plt4.hist(finaldata['Daily Px Change'] * sharesize,bins = 10)
plt4.set(xlabel = 'Daily PnL', ylabel = 'Frequency')
plt4.set_title('DISH Daily PnL Histogram')
#Plt5: DISH Daily Price Change Normal PDF
x = np.linspace(muPxChange - 3 * sigmaPxChange, muPxChange + 3 * sigmaPxChange, 100)
y = stat.norm.pdf(x, muPxChange, sigmaPxChange)
plt5.plot(x,y)
plt5.set(xlabel = 'Daily Price Change', ylabel = 'Probability Density')
plt5.set_title('DISH Daily Price Change Normal PDF')
#Plt6: DISH Daily PnL Normal PDF
x = np.linspace((muPxChange - 3 * sigmaPxChange)*sharesize, (muPxChange + 3 * sigmaPxChange)*sharesize, 100)
y = stat.norm.pdf(x, muPxChange * sharesize, sigmaPxChange*sharesize)
plt6.plot(x,y)
plt6.set(xlabel = 'Daily PnL', ylabel = 'Probability Density')
plt6.set_title('DISH Daily PnL Normal PDF')
plt.show()

#Two stock
rawdata2 = pd.read_csv('CMCSA (6).csv', dtype = object)

#Inspect the data in the dataframe
rawdata2.head()
rawdata2.shape
rawdata2.columns

#Clean up the data
finaldata2 = rawdata2[['Date','Adj Close']]

#Inspect final data
finaldata2.head()
finaldata2.shape
finaldata2.columns

#Calculate daily return for Comcast
AdjCloseFloat2 = finaldata2['Adj Close'].astype(float)
c_dailyreturns = AdjCloseFloat2.pct_change(1)
finaldata2.insert(2,'Daily Return',c_dailyreturns)

#Add a new column to show daily price change
finaldata2['Daily Px Change'] = AdjCloseFloat2 - AdjCloseFloat2.shift(1)

#Drop the first row in the dataframe
finaldata2 = finaldata2.iloc[1:]

#Calculate 2 stock portfolio daily returns
portfolio_return = 0.5 * finaldata['Daily Return'] + 0.5 * finaldata2['Daily Return']

#Calculate 95% historical VaR for the portfolio
portfolioVaRobservation = round(0.05 * len(portfolio_return))
portfoliohistoricalVaR = sorted(portfolio_return)[portfolioVaRobservation]*100
print('The 95% historical VaR is:',portfoliohistoricalVaR,'%')

#Calculate 95% parametric VaR for the portfolio
d_Mu = np.mean(finaldata['Daily Return'])
d_Sigma = np.std(finaldata['Daily Return'])
c_Mu = np.mean(finaldata2['Daily Return'])
c_Sigma = np.std(finaldata2['Daily Return'])
p_Mu = 0.5 * d_Mu + 0.5 * c_Mu
covar = np.cov(finaldata['Daily Return'],finaldata2['Daily Return'])[0,1]
p_Sigma = np.sqrt(((0.5*0.5)*(c_Sigma*c_Sigma))+2*(0.5*0.5)*covar*(c_Sigma*d_Sigma)+(0.5*0.5)*(d_Sigma*d_Sigma))
portfolioParametricVaR = (p_Mu - 1.645 * p_Sigma) * 100
print('The 95% parametric portfolio VaR is:', portfolioParametricVaR, '%')

#Calculate 95% Historical PnL VaR for the portfolio
positionsize = 500000
sharesize_c = positionsize/float(rawdata.iloc[0,5])
sharesize_d = positionsize/float(rawdata2.iloc[0,5])
historicalportfolioPnLVaR = sorted(finaldata['Daily Px Change'])[VaRobservation]*sharesize_d + sorted(finaldata2['Daily Px Change'])[VaRobservation]*sharesize_c
print('The 95% historical portfolio PnL VaR is: $', historicalportfolioPnLVaR)

#Calculate 95% Historical Price Change VaR for the portfolio
historicalportfolioPriceChangeVaR = sorted(finaldata['Daily Px Change'])[VaRobservation] + sorted(finaldata2['Daily Px Change'])[VaRobservation]
print('The 95% historical portfolio Price Change VaR is: $', historicalportfolioPriceChangeVaR)

#Calculate 95% Parametric Price Change VaR for the portfolio
d_Mu = np.mean(finaldata['Daily Px Change'])
d_Sigma = np.std(finaldata['Daily Px Change'])
c_Mu = np.mean(finaldata2['Daily Px Change'])
c_Sigma = np.std(finaldata2['Daily Px Change'])
p_Mu = 0.5 * d_Mu + 0.5 * c_Mu
covar = np.cov(finaldata['Daily Px Change'],finaldata2['Daily Px Change'])[0,1]
p_Sigma = np.sqrt(((0.5*0.5)*(c_Sigma*c_Sigma))+2*(0.5*0.5)*covar*(c_Sigma*d_Sigma)+(0.5*0.5)*(d_Sigma*d_Sigma))
parametricportfolioPxChangeVaR = (p_Mu - 1.645 * p_Sigma)
print('The 95% parametric portfolio Price Change VaR is: $', parametricportfolioPxChangeVaR)

#Calculate 95% Parametric PnL VaR for the portfolio
d_Mu = np.mean(finaldata['Daily Px Change'])
d_Sigma = np.std(finaldata['Daily Px Change'])
c_Mu = np.mean(finaldata2['Daily Px Change'])
c_Sigma = np.std(finaldata2['Daily Px Change'])
p_Mu = 0.5 * d_Mu + 0.5 * c_Mu
covar = np.cov(finaldata['Daily Px Change'],finaldata2['Daily Px Change'])[0,1]
p_Sigma = np.sqrt(((0.5*0.5)*(c_Sigma*c_Sigma))+2*(0.5*0.5)*covar*(c_Sigma*d_Sigma)+(0.5*0.5)*(d_Sigma*d_Sigma))
parametricportfolioPnLVaR = (p_Mu - 1.645 * p_Sigma) * (sharesize_c + sharesize_d)
print('The 95% parametric portfolio PnL VaR is: $', parametricportfolioPnLVaR)


#Create a plot with subplots for 2 stock example
#Portfolio PnL Histogram
fig, (ax1, ax2) =  plt.subplots(2)
fig.suptitle ('Portfolio Daily PnL Analysis', fontsize = 16)
ax1.hist(finaldata['Daily Px Change'] * sharesize_d + finaldata2['Daily Px Change'] * sharesize_c,bins = 10)
ax1.set(xlabel = 'Daily PnL', ylabel = 'Frequency')
ax1.set_title('Portfolio Daily PnL Histogram')
#Portfolio PnL Normal PDF
sharesize = sharesize_c + sharesize_d
x = np.linspace(p_Mu - 3 * p_Sigma * sharesize, p_Mu + 3 * p_Sigma * sharesize, 100)
y = stat.norm.pdf(x, p_Mu * sharesize , p_Sigma * sharesize)
ax2.set(xlabel = 'Daily PnL', ylabel = 'Probability Density')
ax2.set_title('Portfolio Daily PnL Normal PDF')
ax2.plot(x, y)
plt.show()

#Call option prices
#Function to calculate the standard normal CDF (numeric)
def NofX(x):
    return 0.5*(1+math.erf(x/(math.sqrt(2))))

#Function for numerical call price
def CallPrice(S,K,t,sigma,r,q):
    d1 = (math.log(S/K)+t*(r-q+0.5*sigma*sigma))/(sigma*math.sqrt(t))
    d2 = d1 - sigma*math.sqrt(t)
    Nd1 = NofX(d1)
    Nd2 = NofX(d2)
    return S*np.exp(-q*t)*Nd1-K*np.exp(-r*t)*Nd2


sigma = np.std(finaldata['Daily Return'])
callArr = np.array([])
adjcloseArr = np.array(AdjCloseFloat,dtype = float)
i = 0

#Put daily adj close call option prices for DISH in an array
for i in range(len(adjcloseArr)):
    CallPx = CallPrice(adjcloseArr[i],22,1,sigma,0.005,0)
    callArr = np.append(callArr, CallPx)

#Inspect the call option prices array
print(callArr)
callArr.shape

#Calculate daily returns on DISH Call Options
dailyreturnArr = np.array([])
callshiftArr = sp.ndimage.shift(callArr,1)
for i in range(1,len(callArr)):
    dailyreturnCall = (callArr[i] - callshiftArr[i])/callshiftArr[i]
    dailyreturnArr = np.append(dailyreturnArr,dailyreturnCall)

#95% Historical VaR on DISH Call Options
CallVaRObs = round(0.05*(len(dailyreturnArr)))
CallHistoricalVaR = sorted(dailyreturnArr)[CallVaRObs] * 100
print("The 95% historical VaR for DISH call options is:", CallHistoricalVaR, "%")

#95% Parametric VaR on DISH Call Options
muCall = np.mean(dailyreturnArr)
sigmaCall = np.std(dailyreturnArr)
CallParametricVaR = (muCall - 1.645 * sigmaCall) * 100
print("The 95% parametric VaR for DISH call options is:", CallParametricVaR, "%")

#95% Historical PnL VaR on DISH Call Options
CallSize = 1000000/callArr[0]
dailypxchangeArr = np.array([])
for i in range (len(callArr)):
    dailypxchangeCall = (callArr[i] - callshiftArr[i])
    dailypxchangeArr = np.append(dailypxchangeArr,dailypxchangeCall)
CallHistoricalPnLVaR = sorted(dailypxchangeArr)[CallVaRObs] * CallSize
print("The 95% historical PnL VaR for DISH call options is: $",CallHistoricalPnLVaR )

#95% Parametric PnL VaR on DISH Call Options
muCallPxChange = np.mean(dailypxchangeArr)
sigmaCallPxChange = np.std(dailypxchangeArr)
CallParametricPnLVaR = (muCallPxChange - 1.645 * sigmaCallPxChange) * CallSize
print("The 95% parametric PnL VaR for DISH call options is: $", CallParametricPnLVaR)

#Ho-Lee implementation
a1,a2,a3,a4 = 0,0,0,0
tj = np.array([0.08333, 0.16667, 0.25,0.5,1,2,3,5,7,10,20,30])
rates = np.array([0.01,0.03,0.03,0.02,0.11,0.25,0.30,0.41,0.60,0.72,1.09,1.29])
def line (tj,a1,a2,a3,a4):
    return(a1 + a2*tj)*np.exp(-a3*tj) + a4
popt, pcov = sp.optimize.curve_fit(line, tj, rates)
print('The parameters a1 through a4 are as follows: ' , popt)
print('The variance-covariance matrix is as follows: ', pcov)
zj = ([1,2,3,4,5,6,7,8,9,10])
#yield_arr = np.zeros((len(zj),1))
a_yield_arr = np.array(zj,dtype = float)
length_z = len(zj)
count = 0
for i in range (length_z) :
    a_yield_arr[i]= line(zj[i],popt[0], popt[1], popt[2], popt[3])
    count = count + 1
    print('The rate for year ' ,count, 'is: ' , a_yield_arr[i])
    print('The price for year ' ,count, 'is: ' , 100-a_yield_arr[i])


# In[ ]:




