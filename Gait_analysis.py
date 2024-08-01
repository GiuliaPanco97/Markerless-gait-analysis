#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read file
DF_DLC_100 = pd.read_pickle('Your_Path/DLCT_100.pkl')
F=sorted(DF_DLC_100)
set_fps_cam = 25

Time_interval_HeelDx=[]
Time_interval_HeelSx=[]
Time_interval_ToeDx=[]
Time_interval_ToeSx=[]

"""crop video as space coordinates of platform"""
i=0
for i in range(len(DF_DLC_100.columns)):
  Time_interval_HeelDx.append('S'+str(i)+'_T')
  Time_interval_HeelSx.append('S'+str(i)+'_T')
  Time_interval_ToeDx.append('S'+str(i)+'_T')
  Time_interval_ToeSx.append('S'+str(i)+'_T')
z=0
for z in range(len(DF_DLC_100.columns)):
  Time_interval_HeelDx[z]=np.where(((DF_DLC_100.loc[0,F[z]].iloc[2:,19] < 400) & ((DF_DLC_100.loc[0,F[z]].iloc[2:,19]) > 60)))
  Time_interval_ToeDx[z]=np.where(((DF_DLC_100.loc[0,F[z]].iloc[2:,22] < 400) & ((DF_DLC_100.loc[0,F[z]].iloc[2:,22]) > 60)))
z=0
for z in range(len(DF_DLC_100.columns)):
  Time_interval_HeelSx[z]=np.where((DF_DLC_100.loc[0,F[z]].iloc[2:,43] < 400) & ((DF_DLC_100.loc[0,F[z]].iloc[2:,43]) > 60))
  Time_interval_ToeSx[z]=np.where((DF_DLC_100.loc[0,F[z]].iloc[2:,46] < 400) & ((DF_DLC_100.loc[0,F[z]].iloc[2:,46]) > 60))
 
prop_cost=3.1

"""Define empty list"""
Subject=[]
Subject_name=[]
struct_list=[]

"""function"""
def merge(num1, num2):
      arr3 = np.concatenate((num1, num2), axis=None)
      arr3.sort()
      return arr3

def delete_until_nan(arr):
    deleted_length = 0
    for index, element in enumerate(arr):
        if element != element:  # Check for NaN using the fact that NaN != NaN
            deleted_length = index
            return arr[index:], deleted_length
    return arr, deleted_length

def split_array_by_nan_with_index(arr):
  subarrays = []
  subarray = []
  subarray_start = 0
  for i, value in enumerate(arr):
      if np.isnan(value):
          if subarray and len(subarray) >= 20:
              subarrays.append((subarray, subarray_start, i - 1))
          subarray = []
          subarray_start = i + 1
      else:
          subarray.append(value)
  if subarray and len(subarray) >= 20:
      subarrays.append((subarray, subarray_start, len(arr) - 1))
  return subarrays

def filter_subarrays_by_difference(subarrays, threshold):
  filtered_subarrays = []
  for subarray, start_index, end_index in subarrays:
      if abs(subarray[0] - subarray[-1]) >= threshold:
          filtered_subarrays.append((subarray, start_index, end_index))
  return filtered_subarrays

"""Preallocated list"""
for z in range(len(F)):
  Subject.append((F[z]))
  Subject_name.append((F[z]))

"""main loop of code"""
i=0
for i in range(len(F)):
  Data = DF_DLC_100.loc[0,F[i]]
  #print(F[i])
   # insert frames to consider
  fps_initial = 1 # for eventually cutting
  fps_final = len(Data) - 1

  # Define bodypart
  DLCT100_Shoulder_R=[]
  DLCT100_Elbow_R=[]
  DLCT100_Wrist_R=[]
  DLCT100_Hip_R=[]
  DLCT100_Knee_R=[]
  DLCT100_Ankle_R=[]
  DLCT100_Heel_R=[]
  DLCT100_Toe_R=[]
  DLCT100_Shoulder_L=[]
  DLCT100_Elbow_L=[]
  DLCT100_Wrist_L=[]
  DLCT100_Hip_L=[]
  DLCT100_Knee_L=[]
  DLCT100_Ankle_L=[]
  DLCT100_Heel_L=[]
  DLCT100_Toe_L=[]

  DLCT100_Shoulder_R.append(DF_DLC_100.loc[0,F[i]].iloc[2:,1])
  DLCT100_Elbow_R.append(DF_DLC_100.loc[0,F[i]].iloc[2:,4])
  DLCT100_Wrist_R.append(DF_DLC_100.loc[0,F[i]].iloc[2:,7])
  DLCT100_Hip_R.append(DF_DLC_100.loc[0,F[i]].iloc[2:,10])
  DLCT100_Knee_R.append(DF_DLC_100.loc[0,F[i]].iloc[2:,13])
  DLCT100_Ankle_R.append(DF_DLC_100.loc[0,F[i]].iloc[2:,16])
  DLCT100_Heel_R.append(DF_DLC_100.loc[0,F[i]].iloc[2:,19])
  DLCT100_Toe_R.append(DF_DLC_100.loc[0,F[i]].iloc[2:,22])
  DLCT100_Shoulder_L.append(DF_DLC_100.loc[0,F[i]].iloc[2:,25])
  DLCT100_Elbow_L.append(DF_DLC_100.loc[0,F[i]].iloc[2:,28])
  DLCT100_Wrist_L.append(DF_DLC_100.loc[0,F[i]].iloc[2:,31])
  DLCT100_Hip_L.append(DF_DLC_100.loc[0,F[i]].iloc[2:,34])
  DLCT100_Knee_L.append(DF_DLC_100.loc[0,F[i]].iloc[2:,37])
  DLCT100_Ankle_L.append(DF_DLC_100.loc[0,F[i]].iloc[2:,40])
  DLCT100_Heel_L.append(DF_DLC_100.loc[0,F[i]].iloc[2:,43])
  DLCT100_Toe_L.append(DF_DLC_100.loc[0,F[i]].iloc[2:,46])

    #Right
  Shoulder_Dx = np.array(DLCT100_Shoulder_R)
  Shoulder_Dx = Shoulder_Dx.flatten()
  Elbow_Dx = np.array(DLCT100_Elbow_R)
  Elbow_Dx = Elbow_Dx.flatten()
  Wrist_Dx = np.array(DLCT100_Wrist_R)
  Wrist_Dx = Wrist_Dx.flatten()
  Hip_Dx = np.array(DLCT100_Hip_R)
  Hip_Dx = Hip_Dx.flatten()
  Knee_Dx = np.array(DLCT100_Knee_R)
  Knee_Dx = Knee_Dx.flatten()
  Ankle_Dx = np.array(DLCT100_Ankle_R)
  Ankle_Dx = Ankle_Dx.flatten()
  Heel_Dx = np.array(DLCT100_Heel_R)
  Heel_Dx = Heel_Dx.flatten()
  Toe_Dx = np.array(DLCT100_Toe_R)
  Toe_Dx = Toe_Dx.flatten()

    #Left
  Shoulder_Sx = np.array(DLCT100_Shoulder_L)
  Shoulder_Sx = Shoulder_Sx.flatten()
  Elbow_Sx = np.array(DLCT100_Elbow_L)
  Elbow_Sx = Elbow_Sx.flatten()
  Wrist_Sx = np.array(DLCT100_Wrist_L)
  Wrist_Sx = Wrist_Sx.flatten()
  Hip_Sx = np.array(DLCT100_Hip_L)
  Hip_Sx = Hip_Sx.flatten()
  Knee_Sx = np.array(DLCT100_Knee_L)
  Knee_Sx = Knee_Sx.flatten()
  Ankle_Sx = np.array(DLCT100_Ankle_L)
  Ankle_Sx = Ankle_Sx.flatten()
  Heel_Sx = np.array(DLCT100_Heel_L)
  Heel_Sx = Heel_Sx.flatten()
  Toe_Sx = np.array(DLCT100_Toe_L)
  Toe_Sx = Toe_Sx.flatten()

  #define time interval as time in the windows of platform
  Time_Int_HeelDx=Time_interval_HeelDx[i]
  Time_Int_HeelSx=Time_interval_HeelSx[i]
  Time_Int_ToeDx=Time_interval_ToeDx[i]
  Time_Int_ToeSx=Time_interval_ToeSx[i]

  ToT_HeelDx=range(len(Heel_Dx))
  ToT_HeelSx=range(len(Heel_Sx))
  ToT_ToeDx=range(len(Toe_Dx))
  ToT_ToeSx=range(len(Toe_Sx))

  rest_HeelDx=np.setdiff1d(ToT_HeelDx, Time_Int_HeelDx[0])
  rest_HeelSx=np.setdiff1d(ToT_HeelSx, Time_Int_HeelSx[0])
  rest_ToeDx=np.setdiff1d(ToT_ToeDx, Time_Int_ToeDx[0])
  rest_ToeSx=np.setdiff1d(ToT_ToeSx, Time_Int_ToeSx[0])

  Heel_Dx[rest_HeelDx]=np.nan
  Heel_Sx[rest_HeelSx]=np.nan
  Toe_Dx[rest_ToeDx]=np.nan
  Toe_Sx[rest_ToeSx]=np.nan

  heel_strike=abs(Heel_Dx-Heel_Sx)
  toe_off=abs(Toe_Dx-Toe_Sx)
  
    # velocity & acceleration of every step (25 Hz = 0.04 sec)
  velocity_HDx=abs(np.diff(Heel_Dx)/0.04)
  acc_HDx=(abs(np.diff(velocity_HDx)/0.04))
  velocity_HSx=abs(np.diff(Heel_Sx)/0.04)
  acc_HSx=(abs(np.diff(velocity_HSx)/0.04))
  
  velocity_TDx=abs(np.diff(Toe_Dx)/0.04)
  acc_TDx=(abs(np.diff(velocity_TDx)/0.04))
  velocity_TSx=abs(np.diff(Toe_Sx)/0.04)
  acc_TSx=(abs(np.diff(velocity_TSx)/0.04))
  
  """Definition of  gait events"""
  low_limit_HS=0
  high_limit_HS=40
  low_limit_TO=10
  high_limit_TO=100
  threshold_event_TO = 15
  threshold_event_HS = 15
  
  # Find the time points where velocity between the limits
  Heel_strike_Dx_indices = np.where((velocity_HDx > low_limit_HS) & (velocity_HDx < high_limit_HS))[0] 
  Heel_strike_Sx_indices = np.where((velocity_HSx > low_limit_HS) & (velocity_HSx < high_limit_HS))[0] 
  
  toe_off_Dx_indices = np.where((velocity_TDx > low_limit_TO) & (velocity_TDx < high_limit_TO))[0] 
  toe_off_Sx_indices = np.where((velocity_TSx > low_limit_TO) & (velocity_TSx < high_limit_TO))[0] 
     
  def select_value (toe_off_index):
      selected_values = []
      z=0
      for z in range(len(toe_off_index) - 1):
          current_value = toe_off_index[z]
          next_value = toe_off_index[z + 1]
          if abs(current_value - next_value) > threshold_event_TO:
             selected_values.append(current_value)
      selected_values.append(toe_off_index[-1])
      selected_values_array = np.array(selected_values)
      return selected_values_array
  
  def select_value_HS(HS_index):
    selected_values = []
    count = len(HS_index) - 1  # Initialize the countdown
    
    while count > 0:
        current_value = HS_index[count - 1]
        next_value = HS_index[count]
        
        if abs(current_value - next_value) > threshold_event_HS:
            selected_values.append(next_value)
        
        count -= 1  # Decrement the countdown
        
    selected_values.append(HS_index[0])  # Append the first value
    selected_values.reverse()  # Reverse the list to maintain the original order
    
    selected_values_array = np.array(selected_values)
    return selected_values_array
    
  TO_Dx_events= select_value(toe_off_Dx_indices) 
  TO_Sx_events= select_value(toe_off_Sx_indices) 
  HS_Dx_events= select_value_HS(Heel_strike_Dx_indices) 
  HS_Sx_events= select_value_HS(Heel_strike_Sx_indices) 
    
  """calculate gait parameter"""
  #Step length
  step = abs(Heel_Dx[HS_Dx_events]-Heel_Sx[HS_Sx_events])
  
  #STEP TIME
  Step_time=(abs(np.array(HS_Dx_events)-np.array(HS_Sx_events)))/set_fps_cam
  #STANCE TIME
  Peak_tot = merge(HS_Dx_events,HS_Sx_events)
  PeakT_tot = merge(TO_Dx_events,TO_Sx_events)
  PeakT_first=PeakT_tot[::2]
  Peak_first=Peak_tot[::2]
  Stance_time = (abs(PeakT_first - Peak_first))/set_fps_cam
  
  #DOUBLE STEP TIME
  Peak_second=Peak_tot[1::2]
  Double_stance_time = (abs(Peak_second-PeakT_first))/set_fps_cam
  
  """ definizone parametri del cammino"""
  #Gait length (cm)
  Gait_length = step / prop_cost
  
#   #Gait speed (m/s)
  Gait_speed = Gait_length/Step_time
  #Cadence step/min
  Cadence=(1/np.mean(Step_time))*60
  
  """Build the structure with the data """
  #struct list
  Subject_name[i] = {'subject': F[i], 'step (px)':step,'Gait_length(cm)': Gait_length,
                     'Step_time(s)': Step_time, 'Cadence (step/min)': Cadence,
                     'Stance_time':Stance_time, 'Double_stance_time':Double_stance_time,
                     'Gait_speed':Gait_speed}
  
  struct_list.append(Subject_name[i])

# """export to excel"""
#export to excel
df = pd.DataFrame()
df = pd.DataFrame(struct_list)
writer = pd.ExcelWriter("Your_Path/DataGaitDLCT100.xlsx", engine='xlsxwriter')
df.to_excel(writer,  index=False)
writer.save()

df.to_pickle("Your_Path/DataGaitDLCT100.pkl")

