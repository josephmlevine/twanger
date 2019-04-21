#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:07:34 2019

@author: jlevine7
"""

from functions import fft
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig

plt.close('all')


filename = "../data/data_2.lvm"
tSEC, v = pl.loadtxt(filename, skiprows = 24, unpack = True)

ft = np.fft.rfft(v)
db =  20*np.log(np.abs(ft)**2)
freq_arr = np.fft.rfftfreq(len(v), d=1/100)


   
peak_freq_index = sig.argrelmax(db, order =  10)[0]
peak_freqHZ = np.zeros(len(peak_freq_index))
for j in range(len(peak_freq_index)):
     peak_freqHZ[j] = freq_arr[int(peak_freq_index[j])] 


     
fundHZ = peak_freqHZ[0]


# 1 = natural frequency units.
nat_freq_on = 1
nat_amp_on = 1

if nat_freq_on:    
    freq_arr /= fundHZ
    tNAT = tSEC  * fundHZ
    peak_freqNAT = peak_freqHZ/fundHZ
  
# 1 = natural amplitude units, 0 -> dB, 1 -> fundamential = 1 amplitude unit
if nat_amp_on:

    db /= db[peak_freq_index[0]]


#freq = np.fft.rfftfreq(10000)

plt.figure(1)
plt.plot(tSEC, v)


plt.figure(2)
plt.plot(freq_arr,db)
if nat_amp_on:
    plt.ylabel('amplitude(dB/dB_0)')
else:
    plt.ylabel('amplitude(dB)')
if nat_freq_on:
    plt.xlabel('freq(f/f_0)')
else:
    plt.xlabel('freq(Hz)')
