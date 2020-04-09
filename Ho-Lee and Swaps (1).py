#!/usr/bin/env python
# coding: utf-8

# In[66]:


from scipy.stats import norm
import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import math
import scipy as sp

##Ho-Lee implementation

#Yield curve data
a1,a2,a3,a4 = 0,0,0,0
tj = np.array([0.08333, 0.16667, 0.25,0.5,1,2,3,5,7,10,20,30])
rates = np.array([0.01,0.03,0.03,0.02,0.11,0.25,0.30,0.41,0.60,0.72,1.09,1.29])

#Function to fit the line of best fit on the parameters
def line (tj,a1,a2,a3,a4):
    return(a1 + a2*tj)*np.exp(-a3*tj) + a4
popt, pcov = sp.optimize.curve_fit(line, tj, rates)
print('The parameters a1 through a4 are as follows: ' , popt)
print('\nThe variance-covariance matrix is as follows: ', pcov)

#Define an array for the independent variable
zj = ([1,2,3,4,5,6,7,8,9,10])
#yield_arr = np.zeros((len(zj),1))
a_yield_arr = np.array(zj,dtype = float)
length_z = len(zj)
count = 0

#Loop to calculate rates and prices
for i in range (length_z) :
    a_yield_arr[i]= line(zj[i],popt[0], popt[1], popt[2], popt[3])
    count = count + 1
    print('\nThe rate for year ' ,count, 'is: ' , a_yield_arr[i])
    print('The price for year ' ,count, 'is: ' , 100-a_yield_arr[i])

#Define interest rate parameters
r = 0.03
r1 = 0.04
sigma = 0.01

#Define option parameters
opt_mat = 1  
opt_bond_mat = 10  
K = 0.60

#Bond parameters
T = 1
s = 10  
t = 0 
dB = s - T 
dt = 0.1

def bond_price(r,M):
    p = np.exp(-r*M)
    return p

#Bond pricing
ps = bond_price(r,s)
pT = bond_price(r,T)
slope = (math.log(bond_price(r,T+dt))-math.log(bond_price(r,T-dt)))/(2*dt)
lprice = math.log(pT/ps) - dB*(slope) - 0.5*(sigma*sigma)*(T-t)*dB*dB
calc_price = np.exp(lprice)*bond_price(r1,dB)

print('\nThe price of the bond is:',calc_price)

#Euro call option pricing
p_opt = bond_price(r,opt_mat)
p_opt_bond = bond_price(r,opt_bond_mat)
vol = sigma*(dB)*math.sqrt(T-t)
d1 = ((math.log((p_opt_bond)/(K*ps)))/(vol)) + (vol*0.5)
d2 = d1 - vol
call = p_opt_bond*norm.cdf(d1) - K*p_opt*norm.cdf(d2)  
put = p_opt*K*norm.cdf(-d2) - p_opt_bond*norm.cdf(-d1)  
print('The price of the Euro call is: ', call)
    
#Swaps
t_swaps = np.array([1,2,3,5,7,10,20,30])
swap_r = np.array([0.69,0.52,0.51,0.60,0.71,0.81,0.85,0.98])
t_tre = np.array([1,2,3,5,7,10,20,30])
treas_r = np.array([0.11,0.25,0.30,0.41,0.60,0.72,1.09,1.29])


#Function to fit the line of best fit on the parameters
def treas_swap_line (tz,a1,a2,a3,a4):
    return(a1 + a2*tz)*np.exp(-a3*tz) + a4
popt, pcov = sp.optimize.curve_fit(treas_swap_line, t_tre, treas_r)
print('The parameters b1 through b4 are as follows: ' , popt)
print('\nThe variance-covariance matrix is as follows: ', pcov)

#Define an array for the independent variable
zz = ([1,2,3,4,5,6,7,8])
b_yield_arr = np.array(zz,dtype = float)
length_zz = len(zz)
count = 0

#Function to calculate swap price
def swap_price(b_yield_arr):
    discount = 1/(1+b_yield_arr**count)
    end = 1/(1+b_yield_arr**8)
    z = disc_arr = []
    np.append(discount, z)
    swap = (1-end)/(np.sum(disc_arr))
    return swap

#Loop to calculate rates and prices
for i in range (length_zz) :
    b_yield_arr[i]= line(zz[i],popt[0], popt[1], popt[2], popt[3])
    count = count + 1
    print('\nThe rate for year ' ,count, 'is: ' , b_yield_arr[i])
    print('The price for year ' ,count, 'is: ' , 100-b_yield_arr[i])
    print('The swap price for year' , count, 'is: ', swap_price(b_yield_arr[i]))
    z = swap_r[i]-swap_price(b_yield_arr[i])
    if (z!=0):
        print("There is swap arbitrage as follows: ", z)


# In[ ]:




