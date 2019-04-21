#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 20:19:18 2019

@author: jlevine7
"""
import scipy.integrate as spin
import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, sinh, cosh
plt.close('all')

r = 2

def det(a):
    alph = cos(a) + cosh(a) 
    beta = sin(a) + sinh(a)
    gama =  sin(a) - sinh(a) + a*r * (cos(a) - cosh(a))
    delt = -cos(a) - cosh(a) + a*r * (sin(a) - sinh(a))

    return (alph * delt - beta * gama) / cosh(a)

def det_simp(a): #simplify by deviding through by cosh
    alph = cos(a) + 1
    beta = sin(a) + sinh(a)/cosh(a)
    gama =  sin(a) - 1 + a*r * (cos(a) - 1)
    delt = -cos(a) - 1 + a*r * (sin(a) - 1)
    return (alph * delt - beta * gama) 

"""
def u(a_index, x):
    a = roots_arr[a_index]  
    A = -sin(a) * sinh(a) / (cos(a) *cosh(a))
    B = cos(a) / cosh(a) + 1 
    return A * cos(a*x) + sin(a*x) - A * cosh(a*x) - sinh(a*x)
"""


def u(a_index, x):
    a = roots_arr[a_index]  
    A = - (sin(a) + sinh(a))
    B =  cos(a) + cosh(a) 
    return (A/B) * cos(a*x) + sin(a*x) - (A/B) * cosh(a*x) - sinh(a*x)


#find roots 
stop = 30 #final a value to check
root_temp_arr = np.zeros(stop)
for i in range(stop):
    try:
        root_temp_arr[i] = op.brentq(det,i,i+1)     
    except:
        root_temp_arr[i] = 0
        continue

roots_list = [] #form list of roots
for i in range(len(root_temp_arr)):
    if root_temp_arr[i] == 0:
        continue
    roots_list.append(root_temp_arr[i]) 

roots_arr = np.asarray(roots_list)


#roots_arr is eigenvalues of a
omega_bar_arr = roots_arr**2
omega_nat = omega_bar_arr/omega_bar_arr[0] 





roots_diff_arr = np.zeros(len(roots_arr) - 1)
for i in range(len(roots_arr) - 1):
    roots_diff_arr[i] = roots_arr[i+1] - roots_arr[i]





#normilize
N = np.zeros(len(roots_arr))
for i in range(len(roots_arr)):
    integral, error = spin.quad(lambda x: (u(i, x))**2, 0, 1)
    N[i] = 1/(np.sqrt(integral))


#build solutions
t_f = 10
t_steps = 300
x_steps = 50
time = np.linspace(0,t_f,t_steps)
space = np.linspace(0,1,x_steps)
b_n = np.zeros(10)
b_n[0] = .3
b_n[1] = .1
b_n[2] = .0 
b_n[3] = 0
b_n[4] = 0
b_n[5] = .0
b_n[6] = 0
b_n[7] = 0
b_n[8] = 0
b_n[9] = 0

solution_temp = np.zeros((10,x_steps,t_steps))
for t in range(len(time)):
    for x in range(len(space)):
        for n in range(len(b_n)):
            solution_temp[n,x,t] = b_n[n] * u(n, space[x]) * sin(roots_arr[n] * time[t])

solution = np.sum(solution_temp, axis = 0)




#3/9/19 eigen stuff
def matrix_element(i,j):
    return u(i,1) * (omega_bar_arr[i]**2 + omega_bar_arr[j]**2)/2 * u(j,1)

n = 3
matrix = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        matrix[i,j] = matrix_element(i,j)
        
eigval, eigvec = np.linalg.eig(matrix)


#plot function
if 1:
    start = 1
    num = 1000 #number of points to plot
    out_arr = np.zeros(num)
    out_big_arr = np.zeros(num)
    a_arr = np.linspace(start,stop,num)
    
    for i in range(num):
        out_arr[i] = det(a_arr[i]) 
        out_big_arr[i] = det_simp(a_arr[i])     
    plt.plot(a_arr , out_arr)
    plt.plot(a_arr , out_big_arr)
    plt.plot([0, stop], [0, 0], 'r--')
    plt.scatter(roots_arr,np.zeros(len(roots_arr)), label = 'roots')
    plt.title('')
    plt.xlabel('a')
    plt.ylabel('eigenvalue_equation(a)')
    plt.legend()

#plt diff
if 1:
    plt.figure(2)
    plt.scatter(range(len(roots_diff_arr)),abs(roots_diff_arr - np.pi) + np.pi)
    plt.plot([0, len(roots_diff_arr)], [np.pi, np.pi], 'r--') #line @ y = pi
 
#plot shape
if 1:
    plt.figure(3)
    x = np.linspace(0,1,1000)
    plt.plot(x,N[0] *u(0,x))
    plt.plot(x,N[1] *u(1,x))
    plt.plot(x,N[2] *u(2,x))
    plt.plot(x,N[6] *u(6,x))


























































