# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 19:49:06 2018

@author: joseph levine

This is obsolite. fft_simple is up to date. 4.21.19

We used the output from lab view as out array (lmv file). if you did not you will have to
set your sample time manually

makes sure functions file is in directory

"""

from functions import fft
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
plt.close('all')


#input or hard code file names

sampleHZ = 60 
select_input = 'hard code'
if select_input == 'hard code':
    pickup_number = 1
    filename = "../data/2.13_modes.txt"
else:
    pickup_number = int(input("how many pickups did you use? "))
    if pickup_number == 1:
        filenames_dict["1"] = input("enter file name  ")
    else:
        for i in range(pickup_number):
            filenames_dict["{0}".format(i+1)] = input("enter name  ")

#unpack data,make new dict
skip = 2 #skip 2 lines of meta data
v_dict = {}
tSEC = pl.loadtxt(filename, skiprows = skip, unpack = True, usecols = (0,))
for i in range(pickup_number):
    v_dict["{0}".format(i+1)] = pl.loadtxt(filename, skiprows = skip, unpack = True, usecols = (i + 1,))
    #v_dict["{0}".format(i+1)] = np.sin(20000*np.linspace(0,1000,100000))
start_frame = 200
end_frame = 5000 #len(tSEC)

tSEC = tSEC[start_frame:end_frame]
v_dict['1'] = v_dict['1'][start_frame:end_frame]

#run fft
ft_dict, freq_arr, db_dict = fft(v_dict, sampleHZ, pickup_number)

#generate peak frequencys 
peak_freqHZ_dict = {}

for i in range(pickup_number):     
     peak_freq_index = sig.argrelmax(db_dict["{0}".format(i + 1)], order =  100)[0]
     peak_freq_temp_arr = np.zeros(len(peak_freq_index))
     for j in range(len(peak_freq_index)):
         peak_freq_temp_arr[j] = freq_arr[int(peak_freq_index[j])] 
     peak_freqHZ_dict["{0}".format(i+1)] = peak_freq_temp_arr
     peak_freq_temp_arr= np.zeros(len(peak_freq_index))

     
fundHZ = peak_freqHZ_dict['1'][0]

# 1 = natural frequency units.
if 0:    
    freq_arr =  freq_arr/(fundHZ)
    tNAT = tSEC  * fundHZ
    
# 1 = natural amplitude units, 0 -> dB, 1 -> fundamential = 1 amplitude unit
if 1:
    for i in range(pickup_number):
        amp_temp = db_dict["{0}".format(i + 1)][peak_freq_index[0]]
        db_dict["{0}".format(i + 1)] /= amp_temp
        


"""        
plot stuff
"""     
#dB v freq
if 1:
    plt.figure(1)
    plt.plot(freq_arr, db_dict['1'], "r-")
    plt.xlabel("Freq/Freq of fundamental")
    plt.ylabel("Amplitude (dB/dB fundamental)")   
    
#voltage v time
if 1:
    plt.figure(5)
    plt.plot(tNAT, v_dict['1'], 'b-') #v_dict{'i'} plots pickup i 
    plt.xlabel("Time (in cycles)")
    plt.ylabel("displacement")
    


#frequency time spectrum
if 1:
    plt.figure(9)
    f, t, Sxx = sig.spectrogram(v_dict['1'], fs = sampleHZ , nperseg= 200)
    pl.pcolormesh(t, f, (Sxx))
    pl.colorbar(label = 'amplitude')
    pl.ylabel('Frequency (Hz)')
    pl.xlabel('Time (s)')
    plt.ylim(0,18)
    pl.show()





















        

