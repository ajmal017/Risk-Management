import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt

#Read DISH prices from a csv file and drop into a dataframe
path = r'C:\Users\12672\Documents\Books\Finance\Trading\Python Projects\VaR Project DISH\DISH.csv'
rawdata = pd.read_csv(path, dtype = object)

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
print(dailyreturns)

finaldata.insert(2,"Daily Return",dailyreturns)

#Add a new column to show daily price change
finaldata['Daily Px Change'] = AdjCloseFloat - AdjCloseFloat.shift(1)

#Drop first row in daily % return
finaldata = finaldata.iloc[1:]
finaldata.head()
finaldata.shape

#Plot a histogram of daily returns
plt.hist(finaldata['Daily Return'],bins = 10)
plt.xlabel('Daily % Return')
plt.ylabel('Frequency')

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
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
y = stat.norm.pdf(x, mu, sigma)
plt.plot(x, y)
plt.show()

#Draw a daily price change PnL histogram
plt.hist(finaldata['Daily Px Change'],bins = 10)
plt.xlabel('Daily Price Change')
plt.ylabel('Frequency')

#Draw a position PnL histogram
positionsize = 1000000
sharesize = positionsize/float(rawdata.iloc[0,5])
plt.hist(finaldata['Daily Px Change'] * sharesize,bins = 10)
plt.xlabel('Daily PnL')
plt.ylabel('Frequency')

#Historical and Parametric 95% VaR PnL
historicalPxChangeVaR = sorted(finaldata['Daily Px Change'])[VaRobservation]*sharesize
muPxChange = np.mean(finaldata['Daily Px Change'])
sigmaPxChange = np.std(finaldata['Daily Px Change'])
parametricPxChangeVaR = (muPxChange - 1.645 * sigmaPxChange) *sharesize
print('The 95% historical VaR PnL is: $', historicalPxChangeVaR)
print('The 95% parametric VaR PnL is: $', parametricPxChangeVaR)

#Normal PDF Price Change
x = np.linspace(muPxChange - 3 * sigmaPxChange, muPxChange + 3 * sigmaPxChange, 100)
y = stat.norm.pdf(x, muPxChange, sigmaPxChange)
plt.plot(x, y)
plt.show()

#Normal PDF PnL
x = np.linspace((muPxChange - 3 * sigmaPxChange)*sharesize, (muPxChange + 3 * sigmaPxChange)*sharesize, 100)
y = stat.norm.pdf(x, muPxChange * sharesize, sigmaPxChange*sharesize)
plt.plot(x, y)
plt.show()

#Two stock example
path2 = r'C:\Users\12672\Documents\Books\Finance\Trading\Python Projects\VaR Project DISH\CMCSA.csv'
rawdata2 = pd.read_csv(path2, dtype = object)

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

#Portfolio PnL Histogram
plt.hist(finaldata['Daily Px Change'] * sharesize_d + finaldata2['Daily Px Change'] * sharesize_c,bins = 10)
plt.xlabel('Daily PnL')
plt.ylabel('Frequency')

