#SuperRes.py
#Jacob Child
#March 12th, 2025


#%% Packages 
#! .venv\Scripts\Activate.ps1
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
# plt.figure()
# plt.pcolormesh(lfx, lfy, np.sqrt(lfu**2 + lfv**2), cmap=cm.coolwarm, vmin=0.0, vmax=1.0)
# plt.title('Low Res Umag')
# plt.colorbar()

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

# %% Define the model
class SuperRes(nn.Module):
    def __init__(self, n_inchan, n_outchan, ny_up, nx_up):
        super(SuperRes, self).__init__()
        
        self.MySmallUpSample = nn.Upsample(size=(ny_up-2, nx_up-2), mode='bicubic') # expects batch x channels x height x width
        self.MyUpSample = nn.Upsample(size=(ny_up, nx_up), mode='bicubic') # expects batch x channels x height x width
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_inchan, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, n_outchan, 5, padding=2)
        )
        
        #AI recommened weight initalization based off of the paper
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer := layer, nn.Conv2d):
                C_in = layer.in_channels
                bound = (1 / (25 * C_in))**0.5
                nn.init.uniform_(layer.weight, -bound, bound)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
        
    def forward(self, x):
        return set_bcs(self.conv_layers(x))


# see https://en.wikipedia.org/wiki/Finite_difference_coefficient
# or https://web.media.mit.edu/~crtaylor/calculator.html

# f should be a tensor of size: nbatch x nchannels x height (y or eta) x width (x or xi)
# This is written in a general way if one had more data, but for this case there is only 1 data sample, and there are only a few channels it might be clearer to you to separate the channels out into separate variables, in which case the below could be simplified (i.e., you remove the first two dimensions from everything so that input is just height x width if you desire).
def ddxi(f, h):
    # 5-pt stencil
    dfdx_central = (f[:, :, :, 0:-4] - 8*f[:, :, :, 1:-3] + 8*f[:, :, :, 3:-1] - f[:, :, :, 4:]) / (12*h)
    # 1-sided 4pt stencil
    dfdx_left = (-11*f[:, :, :, 0:2] + 18*f[:, :, :, 1:3] -9*f[:, :, :, 2:4] + 2*f[:, :, :, 3:5]) / (6*h)
    dfdx_right = (-2*f[:, :, :, -5:-3] + 9*f[:, :, :, -4:-2] -18*f[:, :, :, -3:-1] + 11*f[:, :, :, -2:]) / (6*h)

    return torch.cat((dfdx_left, dfdx_central, dfdx_right), dim=3)

def ddeta(f, h):
    # 5-pt stencil
    dfdy_central = (f[:, :, 0:-4, :] - 8*f[:, :, 1:-3, :] + 8*f[:, :, 3:-1, :] - f[:, :, 4:, :]) / (12*h)
    # 1-sided 4pt stencil
    dfdy_bot = (-11*f[:, :, 0:2, :] + 18*f[:, :, 1:3, :] -9*f[:, :, 2:4, :] + 2*f[:, :, 3:5, :]) / (6*h)
    dfdy_top = (-2*f[:, :, -5:-3, :] + 9*f[:, :, -4:-2, :] -18*f[:, :, -3:-1, :] + 11*f[:, :, -2:, :]) / (6*h)

    return torch.cat((dfdy_bot, dfdy_central, dfdy_top), dim=2)

def set_bcs(hr_out):
    # bc's
    #The bottom edge (η = 0) is inflow with conditions: u=0,v=1,dp/dη=0. So if u, v, p were tensors of size neta x nxi we would set: u[0, :] = 0; v[0, :] = 1; p[0, :] = p[1, :]. The latter forces the pressure gradient to be zero at the inlet (which just means it is at some unknown constant pressure). The left and right edges are walls with conditions: u=0,v=0,dp/dξ=0 (the latter is a result from incompressible boundary layer theory). At the top (outlet) we set du/dη=0,dv/dη=0,p=0 (the outlet pressure is unknown, but pressure is only defined relative to a reference point so we arbitrarily choose the outlet as a zero reference).
    
    hr_out[:, 2, 0, :] = hr_out[:, 2, 1, :] #p bottom (inlet), deltaP between rows is 0, dp/deta = 0
    hr_out[:, 0, :, 0] = 0.0 #u left (wall)
    hr_out[:, 1, :, 0] = 0.0 #v left (wall)
    hr_out[:, 2, :, 0] = hr_out[:, 2, :, 1] #p left (wall), dp/dxi = 0
    hr_out[:, 0, :, -1] = 0.0 #u right (wall)
    hr_out[:, 1, :, -1] = 0.0 #v right (wall)
    hr_out[:, 2, :, -1] = hr_out[:, 2, :, -2] #p right (wall), dp/dxi = 0
    hr_out[:, 0, -1, :] = hr_out[:, 0, -2, :] #u du/deta = 0
    hr_out[:, 1, -1, :] = hr_out[:, 1, -2, :] #v, dv/deta = 0
    hr_out[:, 2, -1, :] = 0.0 #p
    hr_out[:, 0, 0, :] = 0.0 #u bottom (inlet)
    hr_out[:, 1, 0, :] = 1.0 #v bottom (inlet)
    # print('stop here')
    return hr_out

# in loss-> in square space change bcs, then do derivatives, then convert to d/dx and calculate losses
def big_lossfunc(modelf, hr0_inf, Jinv, dxdxi, dxdeta, dydxi, dydeta, h, rho, mu):
    nu = mu
    # get the model output
    modelf.train()
    hr_out = modelf(hr0_inf) # 1x3x77x49
    #set boundary conditions
    # hr_out = set_bcs(hr_out)
    # calculate derivatives
    dalldxi = ddxi(hr_out, h)
    d2alldxi2 = ddxi(dalldxi, h)
    dalldeta = ddeta(hr_out, h)
    d2alldeta2 = ddeta(dalldeta, h)
    dudxi = dalldxi[:, 0, :, :]
    d2udxi2 = d2alldxi2[:, 0, :, :]
    dvdxi = dalldxi[:, 1, :, :]
    d2vdxi2 = d2alldxi2[:, 1, :, :]
    dpdxi = dalldxi[:, 2, :, :]
    d2pdxi2 = d2alldxi2[:, 2, :, :]
    dudeta = dalldeta[:, 0, :, :]
    d2udeta2 = d2alldeta2[:, 0, :, :]
    dvdeta = dalldeta[:, 1, :, :]
    d2vdeta2 = d2alldeta2[:, 1, :, :]
    dpdeta = dalldeta[:, 2, :, :]
    d2pdeta2 = d2alldeta2[:, 2, :, :]
    
    # convert to d/dx and d/dy using equation 10a and 10b
    dudx = Jinv * (dudxi*dydeta - dudeta*dydxi)
    d2udx2 = Jinv * (d2udxi2*dydeta - d2udeta2*dydxi)
    dvdx = Jinv * (dvdxi*dydeta - dvdeta*dydxi)
    d2vdx2 = Jinv * (d2vdxi2*dydeta - d2vdeta2*dydxi)
    dudy = Jinv * (dudeta*dxdxi - dudxi*dxdeta)
    d2udy2 = Jinv * (d2udeta2*dxdxi - d2udxi2*dxdeta)
    dvdy = Jinv * (dvdeta*dxdxi - dvdxi*dxdeta)
    d2vdy2 = Jinv * (d2vdeta2*dxdxi - d2vdxi2*dxdeta)
    dpdx = Jinv * (dpdxi*dydeta - dpdeta*dydxi)
    dpdy = Jinv * (dpdeta*dxdxi - dpdxi*dxdeta)
    d2pdx2 = Jinv * (d2pdxi2*dydeta - d2pdeta2*dydxi)
    d2pdy2 = Jinv * (d2pdeta2*dxdxi - d2pdxi2*dxdeta)
        
    # calculate losses
    # continuity equation
    cont_loss = torch.mean((dudx + dvdy)**2)
    # momentum equations
    u = hr_out[:, 0, :, :]
    v = hr_out[:, 1, :, :]
    p = hr_out[:, 2, :, :]
    xmom_loss = torch.mean((u*dudx + v*dudy + dpdx/rho - nu * (d2udx2 + d2udy2))**2)
    ymom_loss = torch.mean((u*dvdx + v*dvdy + dpdy/rho - nu * (d2vdx2 + d2vdy2))**2)
    p_loss = torch.mean((-rho*(d2udx2 + 2. * dudy*dvdx + d2vdy2) - d2pdx2 - d2pdy2)**2)
    
    return cont_loss + xmom_loss + ymom_loss + p_loss
    
    
    


# %% Setup
# givens
rho = 1.0
mu = 0.01 
# data setup 
lr_in = torch.stack([torch.tensor(lfu), torch.tensor(lfv)]).unsqueeze(0).float() #  1x2x14x9
dydxi = torch.tensor(dydxi)
dydeta = torch.tensor(dydeta)
dxdxi = torch.tensor(dxdxi)
dxdeta = torch.tensor(dxdeta)
Jinv = torch.tensor(Jinv)
n_inchan = lr_in.shape[1]
n_outchan = 3 #u, v, p

# model setup
model = SuperRes(n_inchan, n_outchan, ny, nx)
model = torch.load('SavedModels/org_nopress_model_epoch5000') #!make sure to update this when needed
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Check plots, upsampled 
# up_umag = torch.sqrt(lr_in[:, 0, :, :]**2 + lr_in[:, 1, :, :]**2)
# up2 = model.MyUpSample(up_umag.unsqueeze(0)).squeeze(0).squeeze(0)
# plt.figure()
# plt.pcolormesh(hfx, hfy, up2, cmap=cm.coolwarm, vmin=0.0, vmax=1.0)
# plt.title('Bicubic Upsampled Umag')
# plt.colorbar()
# # plt.savefig('bicubic_upsampled.png')
# plt.show()


# %% Training
hr0 = model.MyUpSample(lr_in) #upsampled to 77x49

epochs = 5000
plotter_val = 500
losses = []
model.train()
for e in range(epochs):
    optimizer.zero_grad()
    loss = big_lossfunc(model, hr0, Jinv, dxdxi, dxdeta, dydxi, dydeta, h, rho, mu)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (e+1)%plotter_val == 0:
        print(f'Epoch {e+1}/{epochs}, Loss: {loss.item()}')


# %% Post Processing
#save the model 
torch.save(model, 'SavedModels/org_nopress_plus_presspois1_model_epochplus5000')

# Plot losses on a semilog
plt.figure()
plt.semilogy(losses)
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.savefig('Outputs/org_nopress_plus_presspois1_losses.png') #TODO: save on correct models etc
plt.show()


# Evaluate and plot the model
model.eval()
hr_test = model(hr0)
u = hr_test[0, 0, :, :].detach().numpy()
v = hr_test[0, 1, :, :].detach().numpy()
p = hr_test[0, 2, :, :].detach().numpy()

#umag plots 
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#Predicted umag
umag = np.sqrt(u**2 + v**2)
im1 = axes[0].pcolormesh(hfx, hfy, umag, cmap=cm.coolwarm, vmin=0.0, vmax=1.0)
axes[0].set_title('Predicted Umag') 
fig.colorbar(im1, ax=axes[0])
#Actual umag
hfu = hfdata[7, :, :]
hfv = hfdata[8, :, :]
hfp = hfdata[9, :, :]
hf_umag = np.sqrt(hfu**2 + hfv**2)
im2 = axes[1].pcolormesh(hfx, hfy, hf_umag, cmap=cm.coolwarm, vmin=0.0, vmax=1.0)
axes[1].set_title('Actual Umag')
fig.colorbar(im2, ax=axes[1])
#Predicted umag error
umag_error = (umag - hf_umag) / hf_umag #org_nopress_epoch5000 umag error was 0.48
im3 = axes[2].pcolormesh(hfx, hfy, umag_error, cmap=cm.seismic, vmin=-1.0, vmax=1.0)
axes[2].set_title('Umag Prediction Error (Avg: {:.2f})'.format(np.mean(umag_error)))
fig.colorbar(im3, ax=axes[2])
# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('Outputs/org_nopress_plus_presspois1_epochplus5000_umag.png') #TODO: save on correct models etc
plt.show()

#Pressure Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# Predicted Pressure
im1 = axes[0].pcolormesh(hfx, hfy, p, cmap=cm.coolwarm)
axes[0].set_title('Predicted Pressure')
fig.colorbar(im1, ax=axes[0])
# Actual Pressure
im2 = axes[1].pcolormesh(hfx, hfy, hfp, cmap=cm.coolwarm)
axes[1].set_title('Actual Pressure')
fig.colorbar(im2, ax=axes[1])
# Pressure Prediction Error
pressure_error = (p - hfp) / hfp #org_nopress_epoch5000_ pressure error was -0.66
im3 = axes[2].pcolormesh(hfx, hfy, pressure_error, cmap=cm.seismic, vmin=-1.0, vmax=1.0)
axes[2].set_title('Pressure Prediction Error (Avg: {:.2f})'.format(np.mean(pressure_error)))
fig.colorbar(im3, ax=axes[2])
# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('Outputs/org_nopress_plus_presspois1_epochplus5000_pressure.png') #TODO: save on correct models etc
plt.show()


# %%
