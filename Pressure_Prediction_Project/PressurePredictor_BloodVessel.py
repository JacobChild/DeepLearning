#PressurePredictor.py
#Jacob Child

#Pressure Prediction Project: I previously used HW8, the SuperRes homework to predict a pressure field. It kind of helped the Umag field, but the Pressure field is still quite far off. HW 8 was learning to correct the velocity field and predict a pressure field. I am now going to train a CNN/PINN to just do the pressure field. Instead of boundary conditions I am going to give it collocation points so it is kind of like correllating pitot probe data if I was using data from PIV. 

#%% Packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib import cm
import torch 
from torch import nn
import torch.optim as optim
# %% Define the model
class PressurePredictor(nn.Module):
    def __init__(self, n_inchan, n_outchan, ny_up, nx_up):
        super(PressurePredictor, self).__init__()
        
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
        return self.conv_layers(x)


# see https://en.wikipedia.org/wiki/Finite_difference_coefficient
# or https://web.media.mit.edu/~crtaylor/calculator.html

# f should be a tensor of size: nbatch x nchannels x height (y or eta) x width (x or xi)
# This is written in a general way if one had more data, but for this case there is only 1 data sample, and there are only a few channels it might be clearer to you to separate the channels out into separate variables, in which case the below could be simplified (i.e., you remove the first two dimensions from everything so that input is just height x width if you desire). --Dr. Ning's derivative code
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

# in loss-> in square space change bcs, then do derivatives, then convert to d/dx and calculate losses
def big_lossfunc(modelf, hr0_inf, Jinv, dxdxi, dxdeta, dydxi, dydeta, h, rho, mu, col_l, col_r, col_t, col_b): #for presspois3 was col_v, col_h (they were the pitot probe data in the t)
    nu = mu
    # get the model output
    modelf.train()
    p_out = modelf(hr0_inf) # 1x1x77x49

    # stack the output with the input
    uvp_stacked = torch.cat((hr0_inf, p_out), dim=1) # 1x3x77x49
    
    # calculate derivatives
    dalldxi = ddxi(uvp_stacked, h)
    d2alldxi2 = ddxi(dalldxi, h)
    dalldeta = ddeta(uvp_stacked, h)
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
    # pitot probe loss
    # vertical_loss = torch.mean((p_out[0, 0, vert_col, 24] - col_v)**2) # 24 is the centerish of the domain in x
    # horizontal_loss = torch.mean((p_out[0, 0, 38, horz_col] - col_h)**2) # 38 is the centerish of the domain in y
    left_loss = torch.mean((p_out[0, 0, vert_col2, 0] - col_l)**2) # left boundary
    right_loss = torch.mean((p_out[0, 0, vert_col2, -1] - col_r)**2) # right boundary
    top_loss = torch.mean((p_out[0, 0, 0, horz_col2] - col_t)**2) # top boundary
    bottom_loss = torch.mean((p_out[0, 0, -1, horz_col2] - col_b)**2) # bottom boundary
    # momentum equations
    u = uvp_stacked[:, 0, :, :]
    v = uvp_stacked[:, 1, :, :]
    xmom_loss = torch.mean((u*dudx + v*dudy + dpdx/rho - nu * (d2udx2 + d2udy2))**2)
    ymom_loss = torch.mean((u*dvdx + v*dvdy + dpdy/rho - nu * (d2vdx2 + d2vdy2))**2)
    p_loss = torch.mean((-rho*(d2udx2 + 2. * dudy*dvdx + d2vdy2) - d2pdx2 - d2pdy2)**2)
    # print('rounded losses: ', vertical_loss.item(), horizontal_loss.item(), xmom_loss.item(), ymom_loss.item(), p_loss.item())
    return 100*left_loss + 100*right_loss + 100*top_loss + 100*bottom_loss + xmom_loss + 0.1*ymom_loss + 0.001*p_loss #100*vertical_loss + 100*horizontal_loss + xmom_loss + 0.1*ymom_loss + 0.001*p_loss

#%% Load Data (Mostly Dr. Ning's Code)
# load high resolution data, which serves as input to our model
l1f2 = 'Data/sr_hfdata.npy'
l2f2 = 'Pressure_Prediction_Project/Data/sr_hfdata.npy'
#try and catch for loading data
try:
    hfdata = np.load(l1f2)
except FileNotFoundError:
    hfdata = np.load(l2f2)
        

# load high resolution grids and mapping from low resolution to high resolution grid
# hfdata = np.load("sr_hfdata.npy")
Jinv = hfdata[0, :, :]  # size 77 x 49 (height x width)
dxdxi = hfdata[1, :, :]
dxdeta = hfdata[2, :, :]
dydxi = hfdata[3, :, :]
dydeta = hfdata[4, :, :]
hfx = hfdata[5, :, :]
hfy = hfdata[6, :, :]
hfu = hfdata[7, :, :]
hfv = hfdata[8, :, :]
hfp_compare = hfdata[9, :, :]


ny, nx = hfx.shape  #(77 x 49)
h = 0.01  # grid spacing in high fidelity (needed for derivatives)
# givens
rho = 1.0
mu = 0.01 
# data setup 
hr_in = torch.stack([torch.tensor(hfu), torch.tensor(hfv)]).unsqueeze(0).float() #  1x2x77x49
dydxi = torch.tensor(dydxi)
dydeta = torch.tensor(dydeta)
dxdxi = torch.tensor(dxdxi)
dxdeta = torch.tensor(dxdeta)
Jinv = torch.tensor(Jinv)
n_inchan = hr_in.shape[1]
n_outchan = 1 # p

#%% Collocation/Pitot Probe Points
n_col = 10 # number of collocation points
# do 10 down the center of the domain and 10 across the center of the domain
vert_col = np.linspace(0, ny-1, n_col).astype(int) # 10 points down the center of the domain
horz_col = np.linspace(0, nx-1, n_col).astype(int) # 10 points across the center of the domain
pitot_probe_vert = hfp_compare[vert_col, 24] # 38 is the centerish of the domain in y
pitot_probe_horz = hfp_compare[38, horz_col] # 24 is the centerish of the domain in x
pitot_probe_vert = torch.tensor(pitot_probe_vert).float()
pitot_probe_horz = torch.tensor(pitot_probe_horz).float()

#Now try boundary collocation points, ie, how important are the boundaries to the model? to make it fair, only 5 points on each side
vert_col2 = np.linspace(0, ny-1, 5).astype(int) # 5 points down the left and right boundaries
horz_col2 = np.linspace(0, nx-1, 5).astype(int) # 5 points across the top and bottom boundaries
pitot_probe_left = hfp_compare[vert_col2, 0] # left boundary
pitot_probe_right = hfp_compare[vert_col2, -1] # right boundary
pitot_probe_top = hfp_compare[0, horz_col2] # top boundary
pitot_probe_bottom = hfp_compare[-1, horz_col2] # bottom boundary
pitot_probe_left = torch.tensor(pitot_probe_left).float()
pitot_probe_right = torch.tensor(pitot_probe_right).float()
pitot_probe_top = torch.tensor(pitot_probe_top).float()
pitot_probe_bottom = torch.tensor(pitot_probe_bottom).float()

# %% Model Setup
# model setup
model = PressurePredictor(n_inchan, n_outchan, ny, nx)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#%% Training Loop
epochs = 5000
plotter_val = 500
losses = []
model.train()
for e in range(epochs):
    optimizer.zero_grad()
    loss = big_lossfunc(model, hr_in, Jinv, dxdxi, dxdeta, dydxi, dydeta, h, rho, mu, pitot_probe_left, pitot_probe_right, pitot_probe_top, pitot_probe_bottom)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (e+1)%plotter_val == 0:
        print(f'Epoch {e+1}/{epochs}, Loss: {loss.item()}')

# %% Plotting / Evaluating 
#save the model 
# torch.save(model, 'SavedModels/presspois4_model_epoch10000') #TODO: save on correct models etc

# Plot losses on a semilog
plt.figure()
plt.semilogy(losses)
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
# plt.savefig('Outputs/presspois4_losses_epoch10000.png') #TODO: save on correct models etc
plt.show()


# Evaluate and plot the model
model.eval()
p_test = model(hr_in)
p_test = p_test.squeeze(0).squeeze(0) # remove batch and channel dimensions
#Pressure Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# Predicted Pressure
im1 = axes[0].pcolormesh(hfx, hfy, p_test.detach().numpy(), cmap=cm.coolwarm)
axes[0].set_title('Predicted Pressure')
fig.colorbar(im1, ax=axes[0])
# Add collocation points
# axes[0].plot(hfx[vert_col, 24], hfy[vert_col, 24], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
# axes[0].plot(hfx[38, horz_col], hfy[38, horz_col], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[0].plot(hfx[vert_col2, 0], hfy[vert_col2, 0], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[0].plot(hfx[vert_col2, -1], hfy[vert_col2, -1], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[0].plot(hfx[0, horz_col2], hfy[0, horz_col2], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[0].plot(hfx[-1, horz_col2], hfy[-1, horz_col2], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
# Actual Pressure
im2 = axes[1].pcolormesh(hfx, hfy, hfp_compare, cmap=cm.coolwarm)
axes[1].set_title('Actual Pressure')
fig.colorbar(im2, ax=axes[1])
# Add collocation points
# axes[1].plot(hfx[vert_col, 24], hfy[vert_col, 24], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
# axes[1].plot(hfx[38, horz_col], hfy[38, horz_col], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[1].plot(hfx[vert_col2, 0], hfy[vert_col2, 0], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[1].plot(hfx[vert_col2, -1], hfy[vert_col2, -1], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[1].plot(hfx[0, horz_col2], hfy[0, horz_col2], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[1].plot(hfx[-1, horz_col2], hfy[-1, horz_col2], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
# Pressure Prediction Error
pressure_error = (p_test.detach().numpy() - hfp_compare) / hfp_compare 
im3 = axes[2].pcolormesh(hfx, hfy, pressure_error, cmap=cm.seismic, vmin=-1.0, vmax=1.0)
axes[2].set_title(f'Pressure Prediction Error (Avg: {np.mean(np.abs(pressure_error)):.2f}, Median: {np.median(np.abs(pressure_error)):.2f})')
fig.colorbar(im3, ax=axes[2])
# Add collocation points
# axes[2].plot(hfx[vert_col, 24], hfy[vert_col, 24], 'x', color='white', label='Collocation Points', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
# axes[2].plot(hfx[38, horz_col], hfy[38, horz_col], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[2].plot(hfx[vert_col2, 0], hfy[vert_col2, 0], 'x', color='white', label='Collocation Points (10ct)', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[2].plot(hfx[vert_col2, -1], hfy[vert_col2, -1], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[2].plot(hfx[0, horz_col2], hfy[0, horz_col2], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
axes[2].plot(hfx[-1, horz_col2], hfy[-1, horz_col2], 'x', color='white', path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
# Adjust layout and show the plot
# Add a single legend for all subplots (external)
fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
plt.tight_layout()
# plt.savefig('Outputs/presspois4_epoch10000_pressure.png') #TODO: save on correct models etc
plt.show()

# %%
