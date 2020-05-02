#!/usr/bin/env python
# coding: utf-8

# In[121]:


#Author: Cameron R. Shabahang

#Use bootstrapping (Jarrow-Turnbull model) to compute lambda values annually
#Choose one company of a given period (week) to construct the lambda curve
#Relationship between default probabilities and CDS spreads = Jarrow-Turnbull
import pandas as pd
import numpy as np
import math

#Price positive recovery risky coupon bond
#Assume that recovery is received at T
# B = bond price
# T = expiry
# t = time
# P = risk-free discount factor between now and time T
# Q = survival probability between now and time T
# R is recovery rate that is assumed constant
# n = 1 (year)
# j = 1
# p = s / (1-R) = probability of default in one-period Bernoulli
#lamda (lam) = intensity


# In[ ]:





# In[ ]:





# In[123]:


#Import data
rawdata = pd.read_csv("CDS_GLOBAL_SORT.csv")
rawdata.head()
data1 = rawdata[['BUSINESS_DATE', 'OBLIGATION_ASSETRANK', 'CREDIT_EVENTS', 'ALT_TERM', 
                 'INSTITUTION_NAME','BID', 'OFFER', 'TRADE']]
data1 = data1[data1['INSTITUTION_NAME'].str.contains('Arcelor', na = False)]
data1 = data1[data1['OBLIGATION_ASSETRANK'].str.contains('SEN', na = False)]
data1 = data1[data1['CREDIT_EVENTS'].str.contains('MM14', na = False)]
sum_column = data1["BID"] + data1["OFFER"]
data1["MID"] = sum_column/2
data1 = data1.drop(['OBLIGATION_ASSETRANK', 'CREDIT_EVENTS'], axis = 1)
data1 = data1[~data1['TRADE'].notna()]
data1 = data1.drop(['TRADE'], axis = 1)
week = ['2015-JUL-08']
data1 = data1[data1['BUSINESS_DATE'].isin(week)]
data1 = data1.reset_index(drop = True)
data1


# In[124]:


#Merge lambda and term into the dataframe
r = 0.05
term = data1['ALT_TERM']
P = np.exp(term*-r)
Pd = P.to_frame(name = 'P(t)')
lam = np.repeat(0.01, len(P), axis = 0)
lamd = pd.DataFrame(lam, columns = ['lambda'])
jarrow = pd.merge(Pd, lamd, right_index = True, left_index = True)
jarrow = pd.merge(data1, jarrow,  right_index = True, left_index = True)
jarrow

def surv(lam, term):   
    
    for i in range (len(term)):
        if (i == 0):
            Q[i] = np.exp(-lam[i]*term[i])
        else:
            Q[i] = Q[i-1]*np.exp(-lam[i]*(term[i]-term[i-1]))
    return Q 
jarrow


# In[ ]:





# In[ ]:





# In[126]:


#Calculate Qt
Q_calc = surv(lam,term)
Qc = Q_calc.to_frame(name = 'Q(t)')
jarrow1 = pd.merge(jarrow, Qc,  right_index = True, left_index = True)
jarrow1


# In[166]:


import itertools
import operator
import functools
from itertools import accumulate
lists = Q_calc, P

#Function for sum product for column G, p. 190, Chen
#def sumproduct(*lists):
    #return sum(functools.reduce(operator.mul, data) for data in zip(lists))
#PQ = []
#for i in range(len(P)):
    #PQ.append(sumproduct(Q_calc[i], P[i]))
#print(PQ)

#Function for product of risk-free P and default probability Q
def product(*lists):
    return P*Q_calc
PQ = []
for i in range(len(P)):
    PQ = product(Q_calc[i], P[i])
#Function to perform an accumulation to obtain column I
def add_one_by_one(l):
    new_l = []
    cumsum = 0
    for elt in l:
        cumsum += elt
        new_l.append(cumsum)
    return new_l
premium = add_one_by_one(PQ)
print(premium)
jarrow1['Premium'] = premium
jarrow1


# In[201]:


#Function for dQ
def dQ(Q_calc):   
    diffQ = np.zeros(len(Q_calc))
    for i in range (len(Q_calc)):
        if (i == 0):
            diffQ[i] = 1 - Q_calc[i]
        else:
            diffQ[i] = Q_calc[i-1] - Q_calc[i]
    return diffQ
dQ_f = dQ(Q_calc)
jarrow1['dQ'] = dQ_f
jarrow1


# In[204]:


lists_1 = dQ_f, P
def product(*lists_1):
    return P*dQ_f
PdQ = []
for i in range(len(P)):
    PdQ = product(dQ_f[i], P[i])
prot = add_one_by_one(PdQ)
jarrow1['Protection'] = prot
jarrow1


# In[205]:


jarrow1 = jarrow1.drop(['BID', 'OFFER', 'BUSINESS_DATE'], axis = 1)
jarrow1


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


#Default probability formula
r = 0.05
p = 0
s = 0
R = 0.4
p = s/(1-R)

#Poison assumption
#Assume piecewise flat intensity values
T = np.zeros(360)
t = 0
Q = np.zeros(360)
lam = np.zeros(360)
n = 360
for i in range (n):
    Q = np.exp(-sum(lam[i]*(T[i]-T[i-1])))
    Q [i] = Q[i-1]*np.exp(-lam[i]*(T[i]-T[i-1]))

