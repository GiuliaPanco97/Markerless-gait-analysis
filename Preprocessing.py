""" 
Filling missing values with interpolation.
Filtering the data to remove noise.
Removing outliers based on a threshold.
Saving the cleaned and filtered data for further analysis.
"""    
"""import libraries"""
import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import spatial
from scipy import interpolate
from scipy.signal import butter,filtfilt
from statistics import mean
import math
from scipy import signal
import pickle

# csv path
csv_path_DLCT =  '/your_path/'

# butter_lowpass_filter requirements.
#T = 5.0         # Sample Period
fs = 25      # sample rate, Hz
cutoff = 5      # desired cutoff frequency of the filter, Hz , slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic

#"""# Interpolation function"""
# interpolate to fill nan values
def fill_nan(A):
    inds = np.arange(A.shape[0])
    A=pd.to_numeric(A)
    #A=A.to_numpy()
    good = np.where(np.isfinite(A))
    if(len(good[0]) <= 1):
        return A

    # linearly interpolate and then fill the extremes with the mean (relatively similar to)
    # what kalman does
    f = interpolate.interp1d(inds[good], A[good],kind="linear",bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    B = np.where(np.isfinite(B),B,np.nanmean(B))
    return B

def impute_frames(frames):
    return np.apply_along_axis(fill_nan,arr=frames,axis=0)

#"""# filter functions"""
def bp_filter(x, low_f, high_f, samplerate, plot=False):
    low_cutoff_bp = low_f / (samplerate / 2)
    high_cutoff_bp = high_f / (samplerate / 2)

    [b, a] = signal.butter(4, [low_cutoff_bp, high_cutoff_bp], btype='bandpass')
    x_filt = signal.filtfilt(b, a, x)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#"""# Read CSV"""
#### DLC TRAIN ####
#### 2D #####
# Read CSV files and put in variables
csv_path = csv_path_DLCT
pos_csv=[]

# Import csv files, csv_json = position csv
csv_files = [pos_csv for pos_csv in os.listdir(csv_path) if pos_csv.endswith('.csv')]
print('Found: ',len(csv_files),'csv files')

Data_DLCT=[]
Data_DLCT_name=[]
name_DLCT=[]
for file in csv_files:
  temp_df = pd.read_csv(open(csv_path+file))
  Data_DLCT.append(temp_df)
  Data_DLCT_name.append(file)
  name_DLCT.append(file)

DF_DLCT = (pd.DataFrame(Data_DLCT)).T
DF_DLCT.columns = name_DLCT

#"""#Remove outliers - Accuracy threshold """
# DLCT
# Threshold for accuracy 0.5
for i in range(len(DF_DLCT.columns)):
  for j in range(len((DF_DLCT.iloc[0,i]).iloc[1,:])):
    if DF_DLCT.iloc[0,i].iloc[1,j]=='likelihood':
       for z in range(len(DF_DLCT.iloc[0,i].iloc[2:,j])):
          if z<=1:
            continue
          if z >1:
            if float(DF_DLCT.iloc[0,i].iloc[z,j]) < 0.5:   #z=row j column
              DF_DLCT.iloc[0,i].iloc[z,(j-2):j]=np.nan

for i in range(len(DF_DLCT.columns)):
  for j in range(len((DF_DLCT.iloc[0,i]).iloc[1,:])):
    DF_DLCT.iloc[0,i].iloc[2:,j]= pd.to_numeric(DF_DLCT.iloc[0,i].iloc[2:,j])

#"""# Interpolation """
for i in range(len(DF_DLCT.columns)):
  for j in range(len((DF_DLCT.iloc[0,i]).iloc[1,:])):
    if DF_DLCT.iloc[0,i].iloc[1,j]=='x':
      DF_DLCT.iloc[0,i].iloc[2:,j] = impute_frames(DF_DLCT.iloc[0,i].iloc[2:,j])
      DF_DLCT.iloc[0,i].iloc[2:,j+1] = impute_frames(DF_DLCT.iloc[0,i].iloc[2:,j+1])

#"""# filtered"""
for i in range(len(DF_DLCT.columns)):
  for j in range(len((DF_DLCT.iloc[0,i]).iloc[1,:])):
    if DF_DLCT.iloc[0,i].iloc[1,j]=='x':
      DF_DLCT.iloc[0,i].iloc[2:,j] = butter_lowpass_filter(DF_DLCT.iloc[0,i].iloc[2:,j],cutoff, fs, order)
      DF_DLCT.iloc[0,i].iloc[2:,j+1] = butter_lowpass_filter(DF_DLCT.iloc[0,i].iloc[2:,j+1],cutoff, fs, order)

DF_DLCT.to_csv("/your_path/DLCT_100.csv", index=False)
DF_DLCT.to_pickle("/your_path/DLCT_100.pkl")
