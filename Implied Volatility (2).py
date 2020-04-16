#!/usr/bin/env python
# coding: utf-8

# In[37]:


##Use Yahoo finance to download option data including implied vol and plot the vol surface 

#Import typically used packages
from scipy import optimize
import numpy as np
from scipy.interpolate import griddata
from dateutil.parser import parse
from datetime import datetime
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
from functools import partial
import pandas as pd 

#Import the API
from pandas_datareader.data import Options
import yfinance as yf

#Black-Scholes option pricing function
def CDF(x):
    return float(norm.cdf(x))

def BlackScholes(sigma, type = 'c', S = 100, K = 100, T = 1., r = 0.01):
    try:
        d1 = (log(S/K)+(r+sigma*sigma/2.)*T)/(sigma*sqrt(T))
        d2 = d1-sigma*sqrt(T)
        if type=='c':
            return S*CDF(d1)-x*exp(-r*T)*CDF(d2)
        else:
            return x*exp(-r*T)*CDF(-d2)-S*CDF(-d1)
    except: return 0

#3-D plot utilizing matplotlib
def plot3D(X,Y,Z):
    fig = plt.figure()
    ax = Axes3D(fig, azim = -25, elev = 50)
    ax.plot(X,Y,Z,'o')
    plt.xlabel("expiry")
    plt.ylabel("strike")
    plt.show()

#Function to calculate implied volatility
def calc_impl_vol(price = 5, right = 'c', underlying = 100, strike = 100, time = 1, rf = 0.01, inc = 0.001):
    f = lambda x: BlackScholes(x,type=right,S=underlying,K=strike,T=time,r=rf)-price
    #The best rootfinding routine is considered Brent
    return optimize.brentq(f,0,5)

#Get ticker option chain from the API
def get_surface(ticker):
    from pandas_datareader import data as pdr
    yf.pdr_override() 
    # download dataframe
    jpm = pdr.get_data_yahoo("JPM", start="2020-03-16", end="2020-04-16")
    underlying = (jpm['Adj Close'])
    opt = yf.Ticker("JPM")
    raw = opt.option_chain('2020-04-16')
    #q.reset_index(inplace=True)
    #q.set_index('Symbol',inplace=True)
    #vals = []
    print(raw)
    #q = pd.DataFrame(raw)
    #print(q)
    #Iterate through the DataFrame pulled from the API
    for index, row in raw.iterrows():
            #Midpoint of bid/ask spread
            price = (float(row['ask'])+float(row['bid']))/2.0
            exp_d = (row['contract'] - datetime.now()).days
            exp_s = (row['contract'] - datetime.now()).seconds
            exp = (exp_d*24*3600 + exp_s) / (365*24*3600)
            try:
                #Calculate the implied volatility
                impl = calc_impl_vol(price, 'lastPrice', underlying, float(row['strike']), exp)
                vals.append([exp,float(row['strike']),impl])
            except:
                pass
    vals = array(vals).T
    combine_plots(vals[0],vals[1],vals[2])

#Construct the volatility surface using griddata
def make_surf(X,Y,Z):
    XX,YY = meshgrid(linspace(min(X),max(X),250),linspace(min(Y),max(Y),250))
    ZZ = griddata(array([X,Y]).T,array(Z),(XX,YY), method='linear')
    return XX,YY,ZZ

def plot3D(X,Y,Z,fig,ax):
    ax.plot(X,Y,Z,'o', color = 'red')
    plt.xlabel("expiry")
    plt.ylabel("strike")
    
def mesh_plot2(X,Y,Z,fig,ax):
    XX,YY,ZZ = make_surf(X,Y,Z)
    ax.plot_surface(XX,YY,ZZ, color = 'white')
    ax.contour(XX,YY,ZZ)
    plt.xlabel("expiry")
    plt.ylabel("strike")

#Combine the mesh plot and the 3-D plot
def combine_plots(X,Y,Z):
    fig = plt.figure()
    ax = Axes3D(fig, azim = -29, elev = 50)
    mesh_plot2(X,Y,Z,fig,ax)
    plot3D(X,Y,Z,fig,ax)
    plt.show()

get_surface('JPM')


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




