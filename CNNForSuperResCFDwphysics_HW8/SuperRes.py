#SuperRes.py
#Jacob Child
#March 12th, 2025

#%% Packages 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch 
from torch import nn
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler

#%% Load Data (Dr. Ning's Code)
# load low resolution data, which serves as input to our model
l1f1 = 'Data/sr_lfdata.npy'
l2f1 = 'CNNForSuperResCFDwphysics_HW8/Data/sr_lfdata.npy'
l1f2 = 'Data/sr_hfdata.npy'
l2f2 = 'CNNForSuperResCFDwphysics_HW8/Data/sr_hfdata.npy'
#try and catch for loading data
try:
    lfdata = np.load(l1f1)
    hfdata = np.load(l1f2)
except FileNotFoundError:
    lfdata = np.load(l2f1)
    hfdata = np.load(l2f2)
        
# lfdata = np.load("sr_lfdata.npy")
lfx = lfdata[0, :, :]  # size 14 x 9  (height x width)
lfy = lfdata[1, :, :]
lfu = lfdata[4, :, :]
lfv = lfdata[5, :, :]

# plot the low resolution data (like fig 3a except we are using MRI noise here rather than Gaussian noise so it will look a bit different)
plt.figure()
plt.pcolormesh(lfx, lfy, np.sqrt(lfu**2 + lfv**2), cmap=cm.coolwarm, vmin=0.0, vmax=1.0)
plt.title('Low Res Umag')
plt.colorbar()

# load high resolution grids and mapping from low resolution to high resolution grid
# hfdata = np.load("sr_hfdata.npy")
Jinv = hfdata[0, :, :]  # size 77 x 49 (height x width)
dxdxi = hfdata[1, :, :]
dxdeta = hfdata[2, :, :]
dydxi = hfdata[3, :, :]
dydeta = hfdata[4, :, :]
hfx = hfdata[5, :, :]
hfy = hfdata[6, :, :]

ny, nx = hfx.shape  #(77 x 49)
h = 0.01  # grid spacing in high fidelity (needed for derivatives)

plt.show()
# %%
