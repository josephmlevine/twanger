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

plt.close('all')

t = np.linspace(0,2 * np.pi * 1000, 10000)
v = np.sin(t)

freq = np.fft.rfftfreq(10000)

ft = np.fft.rfft(v)

plt.plot(t, v)
plt.figure(2)
plt.plot(freq, ft)