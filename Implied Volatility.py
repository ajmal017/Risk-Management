#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Use Yahoo finance to download option data including implied vol and plot the vol surface 

from pandas_datareader.data import Options
from dateutil.parser import parse
from datetime import datetime
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
#from implied_vol import BlackScholes
from functools import partial
from scipy import optimize
import numpy as np
from scipy.interpolate import griddata

#query{}.finance.yahoo.com/v7/finance/


def CND(X):
   return float(norm.cdf(X))

def BlackScholes(v,CallPutFlag = 'c',S = 100.,X = 100.,T = 1.,r = 0.01):
 
   try:
      d1 = (log(S/X)+(r+v*v/2.)*T)/(v*sqrt(T))
      d2 = d1-v*sqrt(T)
 
      if CallPutFlag=='c':
         return S*CND(d1)-X*exp(-r*T)*CND(d2)
 
      else:
         return X*exp(-r*T)*CND(-d2)-S*CND(-d1)
 
   except: return 0

def plot3D(X,Y,Z):
   fig = plt.figure()
   ax = Axes3D(fig, azim = -29, elev = 50)
   ax.plot(X,Y,Z,'o')
   plt.xlabel("expiry")
   plt.ylabel("strike")
   plt.show()

def calc_impl_vol(price = 5., right = 'c', underlying = 100., strike = 100., time = 1., rf = 0.01, inc = 0.001):
   f = lambda x: BlackScholes(x,CallPutFlag=right,S=underlying,X=strike,T=time,r=rf)-price
   return optimize.brentq(f,0.,5.)

def get_surf(ticker):
 
   q = Options(ticker, 'yahoo').get_all_data()
   q.reset_index(inplace=True)
   q.set_index('Symbol',inplace=True)
   vals = []
 
   print(q.head())
 
   for index, row in q.iterrows():
      if row['Type'] == 'put':
         underlying = float(row['Underlying_Price'])
         price = (float(row['Ask'])+float(row['Bid']))/2.0
         expd = (row['Expiry'] - datetime.now()).days
         exps = (row['Expiry'] - datetime.now()).seconds
         exp = (expd*24.*3600. + exps) / (365.*24.*3600.)
         try:
            impl = calc_impl_vol(price, 'p', underlying, float(row['Strike']), exp)
            vals.append([exp,float(row['Strike']),impl])
         except:
            pass
 
   vals = array(vals).T
   mesh_plot2(vals[0],vals[1],vals[2])
   #if you want to call the 3d plot above use this code instead:
   #plot3D(vals[0],vals[1],vals[2])
   #if you want to call both plots use this code instead:
   #combine_plots(vals[0],vals[1],vals[2])

def make_surf(X,Y,Z):
   XX,YY = meshgrid(linspace(min(X),max(X),230),linspace(min(Y),max(Y),230))
   ZZ = griddata(array([X,Y]).T,array(Z),(XX,YY), method='linear')
   return XX,YY,ZZ

def plot3D(X,Y,Z,fig,ax):
   ax.plot(X,Y,Z,'o', color = 'pink')
   plt.xlabel("expiry")
   plt.ylabel("strike")
   
def mesh_plot2(X,Y,Z,fig,ax):
 
   XX,YY,ZZ = make_surf(X,Y,Z)
   ax.plot_surface(XX,YY,ZZ, color = 'white')
   ax.contour(XX,YY,ZZ)
   plt.xlabel("expiry")
   plt.ylabel("strike")
 
def combine_plots(X,Y,Z):
   fig = plt.figure()
   ax = Axes3D(fig, azim = -29, elev = 50)
   mesh_plot2(X,Y,Z,fig,ax)
   plot3D(X,Y,Z,fig,ax)
   plt.show()

get_surf('JPM')


# In[ ]:





# In[ ]:





# In[ ]:




