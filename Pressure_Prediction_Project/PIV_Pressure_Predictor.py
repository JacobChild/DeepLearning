#PIV_Pressure_Predictor.py
#Jacob Child
#April 9th, 2025

#HW8_Extender.py took the SuperRes homework and extended it to also predict a pressure field. PressurePredictor_BloodVessel.py took the same problem as HW8, but gave the CNN the high fidelity u and v data and then worked on predicting the pressure field by adding collocation points etc.
#This file PIV_Pressure_Predictor.py will use the same architecture and ideas as the BloodVessel one, but I will give it PIV data given by Nathan Welker and try to predict the pressure field.

#%% Import Needed Packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib import cm
import torch 
from torch import nn
import torch.optim as optim

# %% Data Loading
l1 = 'Data/PIV_Data/'
l2 = 'Pressure_Prediction_Project/Data/PIV_Data/' #this is for the debugger to keep running 
#try and catch for loading the data 
#formats 
#PitotAverage: taken @z = 0.0241m and x = 0:120mm  and y = 0? #TODO: check with Nathan Welker
#PitotAverage: time, x, Pressure (Pa), rpm (no need), zeros (no need)
#SideViewPIVVelocity.txt: x (mm), y(mm), Avg u (m/s), avg v(m/s)
#SideViewPIVPressure.txt: ? Pressure (Pa) (112,154)
# WakeViewPIVVelocity.txt: x (mm), y(mm), Vx (m/s), Vy (m/s), Vz (m/s)(z is out of plane/page)
#SideViewCFD.csv: Pressure (Pa), Vx (m/s), Vy (m/s), Vz (m/s), x (m), y(m), z (m)
#WakeViewCFD.csv: Pressure (Pa), Vx (m/s), Vy (m/s), Vz (m/s), x (m), y(m), z (m)
try:
    pitot_data = torch.tensor(np.loadtxt(l1 + 'PitotAverage.txt', skiprows=1)[:,1:3]).float()
    sv_veloc = torch.tensor(np.loadtxt(l1 + 'SideViewPIVVelocity.txt', skiprows=1, delimiter=';')).float() #Side view PIV velocity data
    sv_nw_pred = torch.tensor(np.loadtxt(l1 + 'SideViewPIVPressure.txt', delimiter=',')).float() #Nathan Welker's pressure prediction (112,154) Pa
    wv_veloc = torch.tensor(np.loadtxt(l1 + 'WakeViewPIVVelocity.txt', skiprows=1, delimiter=';')).float() #Wake view PIV velocity data
    sv_cfd = torch.tensor(np.loadtxt(l1 + 'SideViewCFD.csv', skiprows=1, delimiter=',')).float() #Side view CFD data 
    sv_cfd = sv_cfd[:,[4,5,6,1,2,3,0]] #rearrange the columns to be x,y,z,Vx,Vy,Vz,Pressure
    wv_cfd = torch.tensor(np.loadtxt(l1 + 'WakeViewCFD.csv', skiprows=1, delimiter=',')).float() #Wake view CFD data
    wv_cfd = wv_cfd[:,[4,5,6,1,2,3,0]] #rearrange the columns to be x,y,z,Vx,Vy,Vz,Pressure
    
    
except FileNotFoundError:
    pitot_data = torch.tensor(np.loadtxt(l2 + 'PitotAverage.txt', skiprows=1)[:,1:3]).float()
    sv_veloc = torch.tensor(np.loadtxt(l2 + 'SideViewPIVVelocity.txt', skiprows=1, delimiter=';')).float()
    sv_nw_pred = torch.tensor(np.loadtxt(l2 + 'SideViewPIVPressure.txt', delimiter=',')).float()
    wv_veloc = torch.tensor(np.loadtxt(l2 + 'WakeViewPIVVelocity.txt', skiprows=1, delimiter=';')).float()
    sv_cfd = torch.tensor(np.loadtxt(l2 + 'SideViewCFD.csv', skiprows=1, delimiter=',')).float()
    sv_cfd = sv_cfd[:,[4,5,6,1,2,3,0]]
    wv_cfd = torch.tensor(np.loadtxt(l2 + 'WakeViewCFD.csv', skiprows=1, delimiter=',')).float()
    wv_cfd = wv_cfd[:,[4,5,6,1,2,3,0]]
    
    

######### SIDE VIEW FIRST ##########
# %% Plot Data to make sure I understand it
#side view PIV velocity data, just plot u
plt.figure(figsize=(12,8))
plt.title('Side View PIV Velocity Data')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.scatter(sv_veloc[:,0], sv_veloc[:,1], c=sv_veloc[:,2], cmap='jet')
plt.scatter(pitot_data[:,0], np.ones_like(pitot_data[:,1])*-.0241*1000, c='black', marker='x', label='Pitot Data')
plt.colorbar(label='u (m/s)')
plt.show()

#sv_cfd prep: it is the whole cfd domain, I need to keep just the values where x and z (y in the piv data) are in the range of the PIV data.
xmin = torch.min(sv_veloc[:,0]*.001) #convert mm to m
xmax = torch.max(sv_veloc[:,0]*.001) #convert mm to m
zmin = torch.min(sv_veloc[:,1]*.001) #convert mm to m
zmax = torch.max(sv_veloc[:,1]*.001) #convert mm to m
sv_cfd_reduced = sv_cfd[(sv_cfd[:,0] >= xmin) & (sv_cfd[:,0] <= xmax) & (sv_cfd[:,2] >= zmin) & (sv_cfd[:,2] <= zmax)] #keep only the values where x and z are in the range of the PIV data
#plot sv_cfd u #TODO: convert this to a normal grid spacing. After feeding it to the neural net if I do that.
plt.figure(figsize=(12,8))
plt.title('Side View CFD Velocity Data')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.scatter(sv_cfd_reduced[:,0], sv_cfd_reduced[:,2], c=sv_cfd_reduced[:,-1], cmap='jet')
plt.colorbar(label='Pressure (Pa)')
plt.show()

# plot sv_nw_pred pressure data #TODO: Ask Nathan welker what orientation this should be in
sv_nw_x = np.linspace(xmin, xmax, sv_nw_pred.shape[0]) #112 points in x direction
sv_nw_y = np.linspace(zmin, zmax, sv_nw_pred.shape[1]) #154 points in y direction
sv_nw_x, sv_nw_y = np.meshgrid(sv_nw_x, sv_nw_y) #create a meshgrid for the pressure data
sv_nw_pred_transposed = sv_nw_pred.T #transpose the pressure data to match the meshgrid
plt.figure(figsize=(12,8))
plt.title('Side View PIV Pressure Data')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.scatter(sv_nw_x, sv_nw_y, c=sv_nw_pred_transposed.flatten(), cmap='jet')
plt.colorbar(label='Pressure (Pa)') 
plt.show()



# %%

#%% Code stash for future use?

#take a non-uniform grid and make it uniform? AI recommended 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Original data (irregular grid)
x = sv_veloc[:, 0].numpy()  # x-coordinates (mm)
y = sv_veloc[:, 1].numpy()  # y-coordinates (mm)
z = sv_veloc[:, 2].numpy()  # values to interpolate (e.g., velocity or pressure)

# Define the uniform grid
x_uniform = np.linspace(np.min(x), np.max(x), 200)  # 200 points in x
y_uniform = np.linspace(np.min(y), np.max(y), 200)  # 200 points in y
x_grid, y_grid = np.meshgrid(x_uniform, y_uniform)

# Interpolate onto the uniform grid
z_grid = griddata((x, y), z, (x_grid, y_grid), method='linear')  # Use 'linear', 'nearest', or 'cubic'

# Plot the interpolated data
plt.figure(figsize=(12, 8))
plt.title('Interpolated Data on Uniform Grid')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.pcolormesh(x_grid, y_grid, z_grid, cmap='jet', shading='auto')
plt.colorbar(label='Interpolated Value')
plt.show()
