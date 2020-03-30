#!/usr/bin/env python
# coding: utf-8

# In[20]:


#import csv
#from collections import defaultdict

#columns = defaultdict(list) # each value in each column is appended to a list

#with open('DISH.csv') as f:
    #reader = csv.DictReader(f) # read rows into a dictionary format
    #for row in reader: # read a row as {column1: value1, column2: value2,...}
        #for (k,v) in row.items(): # go over each column name and value 
            #columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k
#f=(columns['Adj Close'])

#print(f)

#import numpy as np
#my_array = np.array(f)


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
from scipy.stats import norm

dish = web.get_data_yahoo("DISH",
                            start = "2019-03-27",
                            end = "2020-03-27")

print(dish.tail())

dish_daily_returns0 = dish['Adj Close'].pct_change()
dish_daily_returns = dish_daily_returns0.iloc[1:]
#normal curve and histogram
mean,std=norm.fit(dish_daily_returns)

#Returns histogram
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 10, 1)
n, bins, patches = ax1.hist(dish_daily_returns)
ax1.set_xlabel('Returns')
ax1.set_ylabel('Frequency')
#Normal return distribution chart historical

#PnL daily histogram
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 10, 1)
n, bins, patches = ax2.hist(pnl)
ax2.set_xlabel('%PnL')
ax2.set_ylabel('Frequency')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
y = norm.pdf(x, mean, std)
plt.plot(x, y)
plt.show()

array_length = len(dish_daily_returns)
print(array_length)
five = array_length*0.05
#print(five)
rounded = round(five)
#print (rounded)
descending = sorted(dish_daily_returns)
#print (descending)
historical = (descending[13]*100)

#Historical 1yr 5% VaR for DISH
print ('The 5% percent historical VaR is: ', historical,'%')
#Parametric 1yr % VaR for DISH
mean = np.mean(dish_daily_returns)
sig = np.std(dish_daily_returns)
parametric = (mean - 1.645*sig)*100
print ('The 5% percent parametric VaR is: ', parametric,'%')
#Historical price change VaR

price_change = historical*sharepx
print ('The 5% historical PnL VaR is: ', price_change)
#Parametric price change VaR
para_price_change = parametric*sharepx
print ('The 5% parametric PnL VaR is: ', para_price_change)

#Price change histogram historical
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 10, 1)
n, bins, patches = ax3.hist(pnl2)
ax3.set_xlabel('Price Change PnL')
ax3.set_ylabel('Frequency')
#Price change normal distribution chart historical
#Price change normal distribution chart parametric 

#Two stock
comcast = web.get_data_yahoo("CMCSA",
                            start = "2019-03-27",
                            end = "2020-03-27")
print(comcast.tail())

c_daily_returns0 = comcast['Adj Close'].pct_change()
c_daily_returns = c_daily_returns0.iloc[1:]
t_returns = 0.5*c_daily_returns+0.5*dish_daily_returns
pnl_total = t_returns*1000000

array_length1 = len(t_returns)
print(array_length1)
five1 = array_length1*0.05
#print(five)
rounded1 = round(five1)
#print (rounded)
descending1 = sorted(t_returns)
#print (descending)
historical1 = (descending1[13]*100)

#Historical 1yr 5% VaR for portfolio
print ('The 5% percent historical VaR is: ', historical1,'%')
#Parametric 1yr % VaR for portfolio
mean_d = np.mean(dish_daily_returns)
sigma_d = np.std(dish_daily_returns)
mean_c = np.mean(c_daily_returns)
sigma_c = np.mean(c_daily_returns)
p_mean = 0.5*mean_d+0.5*mean_c
covar = np.cov(dish_daily_returns,c_daily_returns)
p_sigma = np.sqrt(((0.5*0.5)*(sigma_d*sigma_d))+2*(0.5*0.5)*covar*(sigma_d*sigma_c)+(0.5*0.5)*(sigma_c*sigma_c))
p_parametric= (p_mean - 1.645*p_sigma)*100
print ('The 5% percent parametric VaR is: ', p_parametric,'%')
#Historical price change VaR
positionsize = 500000
sharepx_a = positionsize/float(dish.iloc[0,5])
sharepx_b = positionsize/float(comcast.iloc[0,5])
price_change_a = historical*sharepx_a
price_change_b = historical1*sharepx_b
print ('The 5% historical PnL VaR is: ', price_change_a+price_change_b)
#Parametric price change VaR
para_price_change1 = p_parametric*sharepx
print ('The 5% parametric PnL VaR is: ', para_price_change1)

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


for i in range(len(dish)) : 
    call = CallPrice(dish[i],22,1,sig,0.005,0)
print (z.head())
#call

array_length = len(call)
print(array_length)
five2 = array_length*0.05
#print(five)
rounded2 = round(five2)
#print (rounded)
descending2 = sorted(call)
#print (descending)
historical2 = (descending2[13]*100)

#Historical 1yr 5% VaR for DISH call
print ('The 5% percent historical call VaR is: ', historical2,'%')
#Parametric 1yr % VaR for DISH call
mean2 = np.mean(call)
sig2 = np.std(call)
parametric2 = (mean2 - 1.645*sig2)*100
print ('The 5% percent parametric call VaR is: ', parametric2,'%')
#Historical price change VaR for DISH call
price_change2 = historical2*sharepx
print ('The 5% historical PnL call VaR is: ', price_change2)
#Parametric price change VaR for DISH call
para_price_change2 = parametric2*sharepx
print ('The 5% PnL  call VaR is: ', para_price_change2)


# In[ ]:





# In[ ]:





# In[ ]:




