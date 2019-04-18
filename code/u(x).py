#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 18:39:16 2019

@author: jlevine7
"""

import scipy
import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, sinh, cosh
import numpy.linalg as lin
import numpy.linalg
plt.close('all')



def u(a_index,x):
    a = roots_arr[a_index]  
    A = - sin(a) / sinh(a) + 1
    B = cos(a) / cosh(a) + 1 
    C = -A
    D = -B
    return A*cos(a*x) + B*sin(a*x) + C*cosh(a*x) + D*sinh(a*x)

a = roots_arr[0]
alph = cos(a) + cosh(a) 
beta = sin(a) + sinh(a)
gama =  sin(a) - sinh(a) + a*r * (cos(a) - cosh(a))
delt = -cos(a) - cosh(a) + a*r * (sin(a) - sinh(a))

a = np.array([[alph,beta],[gama,delt]])

w,v = lin.eig(a)



"""
def ABCD(a_index):
    a = roots_arr[a_index]
    alph = cos(a) + cosh(a) 
    beta = sin(a) + sinh(a)
    gama =  sin(a) - sinh(a) + a*r * (cos(a) - cosh(a))
    delt = -cos(a) - cosh(a) + a*r * (sin(a) - sinh(a))   
    
    A = - beta
    B = aplh
    C = -A
    D = -B
    ABCD = np.array([A,B,C,D])
"""
    
    
x = np.linspace(0,1,1000)

plt.plot(x,u(0,x))
plt.plot(x,u(1,x))
plt.plot(x,u(2,x))
#plt.plot(x,u(3,x))



#plt.plot(x,u(1,x)/35)
#plt.plot(x,u(2,x)/800)







