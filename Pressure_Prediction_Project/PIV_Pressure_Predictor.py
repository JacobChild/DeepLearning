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
from findiff import FinDiff

# %% Models and Functions 
class PressurePredictor(nn.Module):
    def __init__(self, n_inchan, n_outchan, ny_up, nx_up):
        super(PressurePredictor, self).__init__()
        
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


def gridify_data(data, add_batch_dim=True): #AI recommended and checked
    """
    Convert an array (or tensor) of shape (pts, n_vars) into a grid.

    Assumes that the first two columns of data are x and y coordinates.
    
    Parameters:
      data (np.ndarray or torch.Tensor): Input data of shape (pts, n_vars).
      add_batch_dim (bool): If True, returns a tensor with an additional leading batch dimension.
                            Default is True.
    
    Returns:
      torch.Tensor: Gridded data with shape (1, n_vars, num_y, num_x) if add_batch_dim is True,
                    or shape (n_vars, num_y, num_x) otherwise.
    """
    # Convert to numpy array if necessary
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    
    # Get unique x and y values (sorted)
    unique_x = np.unique(data[:, 0])
    unique_y = np.unique(data[:, 1])
    num_x = unique_x.shape[0]
    num_y = unique_y.shape[0]

    # Create an empty grid of shape (num_y, num_x, n_vars)
    grid = np.empty((num_y, num_x, data.shape[1]), dtype=data.dtype)

    # For each point, find the indices corresponding to its x and y values
    for point in data:
        x_val, y_val = point[0], point[1]
        i_x = np.searchsorted(unique_x, x_val)
        i_y = np.searchsorted(unique_y, y_val)
        grid[i_y, i_x, :] = point

    # Transpose the grid to shape (n_vars, num_y, num_x)
    grid = grid.transpose(2, 0, 1)

    # Optionally add a batch dimension to get (1, n_vars, num_y, num_x)
    if add_batch_dim:
        grid = np.expand_dims(grid, axis=0)

    # Return as a torch.Tensor for consistency
    return torch.tensor(grid)


### The Derivative Functions are from Dr. Ning ### 
# see https://en.wikipedia.org/wiki/Finite_difference_coefficient
# or https://web.media.mit.edu/~crtaylor/calculator.html

# f should be a tensor of size: nbatch x nchannels x height (y or eta) x width (x or xi)
# This is written in a general way if one had more data, but for this case there is only 1 data sample, and there are only a few channels it might be clearer to you to separate the channels out into separate variables, in which case the below could be simplified (i.e., you remove the first two dimensions from everything so that input is just height x width if you desire). --Dr. Ning's derivative code
def ddx(f, h):
    # 5-pt stencil
    dfdx_central = (f[:, :, :, 0:-4] - 8*f[:, :, :, 1:-3] + 8*f[:, :, :, 3:-1] - f[:, :, :, 4:]) / (12*h)
    # 1-sided 4pt stencil
    dfdx_left = (-11*f[:, :, :, 0:2] + 18*f[:, :, :, 1:3] -9*f[:, :, :, 2:4] + 2*f[:, :, :, 3:5]) / (6*h)
    dfdx_right = (-2*f[:, :, :, -5:-3] + 9*f[:, :, :, -4:-2] -18*f[:, :, :, -3:-1] + 11*f[:, :, :, -2:]) / (6*h)

    return torch.cat((dfdx_left, dfdx_central, dfdx_right), dim=3)

def ddy(f, h):
    # 5-pt stencil
    dfdy_central = (f[:, :, 0:-4, :] - 8*f[:, :, 1:-3, :] + 8*f[:, :, 3:-1, :] - f[:, :, 4:, :]) / (12*h)
    # 1-sided 4pt stencil
    dfdy_bot = (-11*f[:, :, 0:2, :] + 18*f[:, :, 1:3, :] -9*f[:, :, 2:4, :] + 2*f[:, :, 3:5, :]) / (6*h)
    dfdy_top = (-2*f[:, :, -5:-3, :] + 9*f[:, :, -4:-2, :] -18*f[:, :, -3:-1, :] + 11*f[:, :, -2:, :]) / (6*h)

    return torch.cat((dfdy_bot, dfdy_central, dfdy_top), dim=2)

# Functions for Training 
def big_lossfunc(modelf, velocsf, hf, rho, mu, comp_indf, pitot_pressf, loss_fnf, print_flag=False): #for presspois3 was col_v, col_h (they were the pitot probe data in the t)
    # get the model output
    modelf.train()
    p_out = modelf(velocsf) # 1x1x130x170

    # stack the output with the input
    uvp_stacked = torch.cat((velocsf, p_out), dim=1) # 1x3x77x49
    
    # calculate derivatives: dudx, dudy, dvdx, dvdy, dpdx, dpdy, d2udx2, d2udy2, d2pdx2, d2pdy2
    #Dr. Ning's code: needs fixed grid
    dalldx = ddx(uvp_stacked, hf) #1,3,130,172
    d2alldx2 = ddx(dalldx, hf)
    dalldy = ddy(uvp_stacked, hf)
    d2alldy2 = ddy(dalldy, hf)
    dudx = dalldx[:, 0, :, :]
    d2udx2 = d2alldx2[:, 0, :, :]
    dvdx = dalldx[:, 1, :, :]
    d2vdx2 = d2alldx2[:, 1, :, :]
    dpdx = dalldx[:, 2, :, :]
    d2pdx2 = d2alldx2[:, 2, :, :]
    dudy = dalldy[:, 0, :, :]
    d2udy2 = d2alldy2[:, 0, :, :]
    dvdy = dalldy[:, 1, :, :]
    d2vdy2 = d2alldy2[:, 1, :, :]
    dpdy = dalldy[:, 2, :, :]
    d2pdy2 = d2alldy2[:, 2, :, :]
    
    #FinDiff package
    # dx = FinDiff()
    
        
    # calculate losses
    # pitot probe loss
    pressure_field = p_out[0,0,:,:]
    probe_pressures = pressure_field[comp_indf[:,0], comp_indf[:,1]] #get the pressures at the pitot probe locations
    pitot_loss = loss_fnf(probe_pressures, pitot_pressf) #loss between the predicted pressures and the pitot probe pressures
    
    # momentum equations
    u = uvp_stacked[:, 0, :, :]
    v = uvp_stacked[:, 1, :, :]
    xmom_loss = torch.mean((u*dudx + v*dudy + dpdx/rho - mu * (d2udx2 + d2udy2))**2)
    ymom_loss = torch.mean((u*dvdx + v*dvdy + dpdy/rho - mu * (d2vdx2 + d2vdy2))**2)
    p_loss = torch.mean((-rho*(d2udx2 + 2. * dudy*dvdx + d2vdy2) - d2pdx2 - d2pdy2)**2)
    if print_flag:
        print_flag = False
        print('pitot_loss: ', pitot_loss.item(),'xmom_loss:', xmom_loss.item(), 'ymom_loss:', ymom_loss.item(), 'p_loss:', p_loss.item())
    return 100000*pitot_loss + xmom_loss + ymom_loss + p_loss/10000 #100*vertical_loss + 100*horizontal_loss + xmom_loss + 0.1*ymom_loss + 0.001*p_loss

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
    pitot_data = torch.tensor(np.loadtxt(l1 + 'PitotAverage.txt', skiprows=1)[:,1:3]).float() # Pitot probe, x, Pressure (Pa) data
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
    

# %% Plot Data to make sure I understand it
#shift the Side View PIV Data in x 45mm -> 30mm was found to look better 
sv_veloc[:,0] = sv_veloc[:,0] + 30  #shift the x data to be in the same range as the CFD data
#side view PIV velocity data, just plot u
plt.figure(figsize=(12,8))
plt.title('Side View PIV Velocity Data')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.scatter(sv_veloc[:,0], sv_veloc[:,1], c=sv_veloc[:,2], cmap='jet', vmin = torch.min(sv_veloc[:,2]), vmax = torch.max(sv_veloc[:,2]))
# plt.scatter(pitot_data[:,0], np.ones_like(pitot_data[:,1])*-.0241*1000, c='black', marker='x', label='Pitot Data')
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
plt.title('Side View CFD Pressure Data')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.scatter(sv_cfd_reduced[:,0], sv_cfd_reduced[:,2], c=sv_cfd_reduced[:,-1], cmap='jet')
plt.colorbar(label='Pressure (Pa)')
plt.show()

# plot sv_nw_pred pressure data #TODO: Ask Nathan welker what orientation this should be in, ANS: I need to rotate and flip
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

#%% ######### SIDE VIEW FIRST ##########
print('I am running everything on the side view data first')
#%% Data Wrangling 
#Convert the data to the right format for the model, ie a 1 x 2 x num_x, num_y tensor: the following was AI recommended, but found to work
sv_veloc_grid = gridify_data(sv_veloc, add_batch_dim=True) #1 x 4 x num_y (130) x num_x (170) tensor

#Plot the reshaped data to make sure it looks good
plt.figure(figsize=(12, 8))
plt.title('Regridded: Direct Mapping by Coordinates')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
sc = plt.scatter(
    sv_veloc_grid[0, 0, :, :].flatten(),   # x coordinates from channel 0
    sv_veloc_grid[0, 1, :, :].flatten(),   # y coordinates from channel 1
    c=sv_veloc_grid[0, 2, :, :].flatten(),   # u values from channel 2
    cmap='jet'
)
plt.colorbar(sc, label='u (m/s)')
plt.show()

#sv_veloc_grid now has shape (1, 4, num_y, num_x) where 4 is the number of channels (x,y,u,v) and 1 is the batch size.
#%% Pitot Probe Comparison data Wrangling (AI recommended, my code was originally for the array version)
# === Step 1: Extract x and y grids from sv_veloc_grid ===
x_grid = sv_veloc_grid[0, 0, :, :]  # shape: (num_y, num_x)
y_grid = sv_veloc_grid[0, 1, :, :]  # shape: (num_y, num_x)

# === Step 2: Find the grid row (y index) closest to pitot y ===
pitot_y = -24.1  # in mm
y_array = y_grid[:, 0]  # assume one unique y value per row
abs_diffs = torch.abs(y_array - pitot_y)
min_y_diff = torch.min(abs_diffs)
valid_y_indices = torch.where(torch.abs(y_array - pitot_y) == min_y_diff)[0]
valid_y_index = valid_y_indices[0]   # choose the first match
print("Chosen y row index:", valid_y_index.item(), "with y =", y_array[valid_y_index].item())

# === Step 3: Identify x indices within pitot x-range ===
pitot_xmin = torch.min(pitot_data[:, 0])
pitot_xmax = torch.max(pitot_data[:, 0])
x_in_range = (x_grid[valid_y_index, :] >= pitot_xmin) & (x_grid[valid_y_index, :] <= pitot_xmax)
valid_x_indices = torch.where(x_in_range)[0]
print("Found", valid_x_indices.numel(), "valid x indices in the range.")

# === Step 4: Build final indices and extract filtered data ===
# Create a list of valid (row, col) grid indices:
valid_indices = [(valid_y_index.item(), int(x_idx.item())) for x_idx in valid_x_indices]
print("Valid grid indices (row, col):", valid_indices)

# Extract the filtered velocity data from sv_veloc_grid.
# The result will have shape (channels, number of valid points).
filtered_sv_veloc = sv_veloc_grid[0, :, valid_y_index, valid_x_indices]
print("Filtered velocity data shape:", filtered_sv_veloc.shape)

# Optionally, plot these filtered points to check:
plt.figure(figsize=(12, 8))
plt.title("Filtered SV Velocity Data (Pitot Region)")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
sc = plt.scatter(
    sv_veloc_grid[0, 0, :, :].flatten(),  # grid x values
    sv_veloc_grid[0, 1, :, :].flatten(),  # grid y values
    c=sv_veloc_grid[0, 2, :, :].flatten(),  # grid u values
    cmap="jet"
)
# Overplot the filtered points in a contrasting color:
plt.scatter(
    x_grid[valid_y_index, valid_x_indices].flatten(),
    y_grid[valid_y_index, valid_x_indices].flatten(),
    c="k",
    marker="o",
    label="Pitot region"
)
plt.legend()
plt.colorbar(sc, label="u (m/s)")
plt.show()

#reduce the pitot_data size to match valid_x_indices, ie 83 points, so average every 723 points
pitot_pressures_reduced = torch.zeros(len(valid_x_indices))
for i in range(len(valid_x_indices)):
    pitot_pressures_reduced[i] = torch.mean(pitot_data[723*i:723*(i+1), 1]) #average the pressures over the 723 points
valid_indices = np.array(valid_indices) #convert to numpy array for easier indexing

#%% CFD Comparison Data wrangling
#find the indices of sv_veloc_grid that match (or are closest to) the x,z coordinates of sv_cfd_reduced
#sv_cfd_reduced is in the format x,y,z,Vx,Vy,Vz,Pressure


#%% Setup the model and training
sv_just_veloc = sv_veloc_grid[:,2:,:,:] #1 x 2 x num_y (130) x num_x (172) tensor
in_chan = 2 #2 velocity components
out_chan = 1 #pressure component
num_y = sv_just_veloc.shape[2] #number of y points (130)
num_x = sv_just_veloc.shape[3] #number of x points (172)

model = PressurePredictor(in_chan, out_chan, num_y, num_x) #1 x 2 x num_y (130) x num_x (172) tensor
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#%% Training Loop
rho = 1.225 #kg/m^3
mu = 1.823e-5 #Pa*s at apx 72deg F
h = (torch.max(x_grid) - torch.min(x_grid)) / num_x #grid spacing in x direction (mm)
h = h * 0.001 #convert to m
epochs = 5000
plotter_val = 500
losses = []
model.train()
for e in range(epochs):
    optimizer.zero_grad()
    loss = big_lossfunc(model, sv_just_veloc, h, rho, mu, valid_indices, pitot_pressures_reduced, loss_fn)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (e+1)%plotter_val == 0:
        print(f'Epoch {e+1}/{epochs}, Loss: {loss.item()}')
        big_lossfunc(model, sv_just_veloc, h, rho, mu, valid_indices, pitot_pressures_reduced, loss_fn, print_flag=True)


# %% Plotting / Evaluating 
#TODO: 4 things are being saved, comment or uncomment **ALL** of them
#save the model 
# torch.save(model, 'SavedModels/PIV_sv_init2_epoch5000') #TODO: save on correct models etc

# Plot losses on a semilog
plt.figure()
plt.semilogy(losses)
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
# plt.savefig('Outputs/PIV_sv_init2_losses_epoch5000.png') #TODO: save on correct models etc
plt.show()

#Evaluate the model
model.eval()
p_out_sv = model(sv_just_veloc) #1 x 1 x num_y (130) x num_x (170) tensor
p_out_sv = p_out_sv.squeeze(0).squeeze(0) #remove the batch and channel dimensions
p_out_sv_np = p_out_sv.detach().numpy() #convert to numpy for plotting
max_p_sv = torch.max(p_out_sv) #get the max pressure value
min_p_sv = torch.min(p_out_sv) #get the min pressure value

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# Plot the model output pressure field
axes[0].set_title('Model Output Pressure Field')
axes[0].set_xlabel('x (mm)')
axes[0].set_ylabel('y (mm)')
sc1 = axes[0].scatter(
    x_grid.flatten(), y_grid.flatten(),
    c=p_out_sv_np.flatten(), cmap='jet', vmin=min_p_sv, vmax=max_p_sv
)
fig.colorbar(sc1, ax=axes[0], label='Pressure (Pa)')
# Plot the CFD data pressure field
axes[1].set_title('CFD Pressure Field')
axes[1].set_xlabel('x (m)')
axes[1].set_ylabel('z (m)')
sc2 = axes[1].scatter(
    sv_cfd_reduced[:, 0], sv_cfd_reduced[:, 2],
    c=sv_cfd_reduced[:, -1], cmap='jet', vmin=min_p_sv, vmax=max_p_sv
)
fig.colorbar(sc2, ax=axes[1], label='Pressure (Pa)')
# Adjust layout and save the figure
plt.tight_layout()
# plt.savefig('Outputs/PIV_sv_init2_pressures.png')  # Save the figure
plt.show()

#pitot probe comparison
pitot_x_reduced = x_grid[valid_y_index, valid_x_indices].flatten()
plt.figure(figsize=(12, 8))
plt.title('Pitot Probe Pressure Comparison')
plt.xlabel('x (mm)')
plt.ylabel('Pressure (Pa)')
plt.scatter(pitot_data[:, 0], pitot_data[:, 1], c='black', marker='x', label='Originial Pitot Data', alpha=0.25)
plt.scatter(pitot_x_reduced, pitot_pressures_reduced, label='Reduced Pitot Data', c ='#00FFFF', marker = 's')
plt.scatter(pitot_x_reduced, p_out_sv_np[valid_indices[:, 0], valid_indices[:, 1]], c='red', label='Model Output')
plt.legend()
# plt.savefig('Outputs/PIV_sv_init2_pitot.png') #TODO: save on correct models etc
plt.show()


#%% ######### WAKE VIEW ##########
#Functions 

def ddz(f, h):
    # 5-pt stencil
    dfdz_central = (f[:, 0:-4, :, :] - 8*f[:, 1:-3, :, :] + 8*f[:, 3:-1, :, :] - f[:, 4:, :, :]) / (12*h)
    # 1-sided 4pt stencil
    dfdz_bot = (-11*f[:, 0:2, :, :] + 18*f[:, 1:3, :, :] -9*f[:, 2:4, :, :] + 2*f[:, 3:5, :, :]) / (6*h)
    dfdz_top = (-2*f[:, -5:-3, :, :] + 9*f[:, -4:-2, :, :] -18*f[:, -3:-1, :, :] + 11*f[:, -2:, :, :]) / (6*h)

    return torch.cat((dfdz_bot, dfdz_central, dfdz_top), dim=1)
    
    
def big_lossfunc_3D(modelf, velocsf, hf, rho, mu, comp_indf, pitot_pressf, loss_fnf, print_flag=False): #for presspois3 was col_v, col_h (they were the pitot probe data in the t)
    # get the model output
    modelf.train()
    p_out = modelf(velocsf) # 1x1x130x170

    # stack the output with the input
    uvwp_stacked = torch.cat((velocsf, p_out), dim=1) # 1x3x77x49
    
    # calculate derivatives: dudx, dudy, dvdx, dvdy, dpdx, dpdy, d2udx2, d2udy2, d2pdx2, d2pdy2
    #Dr. Ning's code: needs fixed grid
    dalldx = ddx(uvwp_stacked, hf) #1,4,130,172
    d2alldx2 = ddx(dalldx, hf)
    dalldy = ddy(uvwp_stacked, hf)
    d2alldy2 = ddy(dalldy, hf)
    dalldz = ddz(uvwp_stacked, hf) #1,4,130,172
    d2alldz2 = ddz(dalldz, hf)
    #dx
    dudx = dalldx[:, 0, :, :]
    d2udx2 = d2alldx2[:, 0, :, :]
    dvdx = dalldx[:, 1, :, :]
    d2vdx2 = d2alldx2[:, 1, :, :]
    dwdx = dalldx[:, 2, :, :]
    d2wdx2 = d2alldx2[:, 2, :, :]
    dpdx = dalldx[:, 3, :, :]
    d2pdx2 = d2alldx2[:, 3, :, :]
    #dy
    dudy = dalldy[:, 0, :, :]
    d2udy2 = d2alldy2[:, 0, :, :]
    dvdy = dalldy[:, 1, :, :]
    d2vdy2 = d2alldy2[:, 1, :, :]
    dwdy = dalldy[:, 2, :, :]
    d2wdy2 = d2alldy2[:, 2, :, :]
    dpdy = dalldy[:, 3, :, :]
    d2pdy2 = d2alldy2[:, 3, :, :]
    #dz
    dudz = dalldz[:, 0, :, :]
    d2udz2 = d2alldz2[:, 0, :, :]
    dvdz = dalldz[:, 1, :, :]
    d2vdz2 = d2alldz2[:, 1, :, :]
    dwdz = dalldz[:, 2, :, :]
    d2wdz2 = d2alldz2[:, 2, :, :]
    dpdz = dalldz[:, 3, :, :]
    d2pdz2 = d2alldz2[:, 3, :, :]
        
    # calculate losses
    # pitot probe loss
    pressure_field = p_out[0,0,:,:]
    probe_pressures = pressure_field[comp_indf[:,0], comp_indf[:,1]] #get the pressures at the pitot probe locations
    pitot_loss = loss_fnf(probe_pressures, pitot_pressf) #loss between the predicted pressures and the pitot probe pressures
    
    # momentum equations
    u = uvwp_stacked[:, 0, :, :]
    v = uvwp_stacked[:, 1, :, :]
    w = uvwp_stacked[:, 2, :, :]
    cont_loss = torch.mean((dudx + dudy + dwdz)**2) #continuity equation #TODO: I took this out?
    xmom_loss = torch.mean((u*dudx + v*dudy + w*dudz + dpdx/rho - mu * (d2udx2 + d2udy2 + d2udz2))**2)
    ymom_loss = torch.mean((u*dvdx + v*dvdy + w*dvdz + dpdy/rho - mu * (d2vdx2 + d2vdy2 + d2vdz2))**2)
    zmom_loss = torch.mean((u*dwdx + v*dwdy + w*dwdz + dpdz/rho - mu * (d2vdz2 + d2udz2 + d2wdz2))**2)
    p_loss = torch.mean((-rho*(d2udx2 + 2. * dudy*dvdx + d2vdy2) - d2pdx2 - d2pdy2)**2)
    
    if print_flag:
        print_flag = False
        print('pitot_loss: ', pitot_loss.item(),'xmom_loss:', xmom_loss.item(), 'ymom_loss:', ymom_loss.item(), 'zmom_loss: ', zmom_loss.item(), p_loss:', p_loss.item())
    return 100000*pitot_loss + xmom_loss + ymom_loss + zmom_loss p_loss/10000 #100*vertical_loss + 100*horizontal_loss + xmom_loss + 0.1*ymom_loss + 0.001*p_loss
    
#%% Data Wrangling for Wake View
#wv_veloc is in the format (pts,5) where the 5 is x,y,Vx,Vy,Vz
#wv_cfd is (pts,7) where the 7 is x,y,z,Vx,Vy,Vz,Pressure
wv_veloc_grid = gridify_data(wv_veloc, add_batch_dim=True) #1 x 5 x num_y (130) x num_x (170) tensor
#plot the wv_veloc u and the wv_cfd u to make sure they are the same
wv_umin = torch.min(wv_veloc_grid[0,2,:,:]) #get the min velocity value
wv_umax = torch.max(wv_veloc_grid[0,2,:,:]) #get the max velocity value
plt.figure(figsize=(12,8))
plt.title('Wake View Velocity Data')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.scatter(wv_veloc_grid[0, 0, :, :].flatten(), wv_veloc_grid[0, 1, :, :].flatten(), c=wv_veloc_grid[0, 2, :, :].flatten(), cmap='jet', vmin = wv_umin, vmax = wv_umax)
plt.colorbar(label='u (m/s)')
plt.show()

#wv_cfd prep: it is the whole cfd domain, I need to keep just the values where x and y are in the range of the PIV data.
wv_xmin = torch.min(wv_veloc[:,0]*.001) #convert mm to m
wv_xmax = torch.max(wv_veloc[:,0]*.001) #convert mm to m
wv_ymin = torch.min(wv_veloc[:,1]*.001) #convert mm to m
wv_ymax = torch.max(wv_veloc[:,1]*.001) #convert mm to m
zloc = 0.0241 #m, this is the z location of the PIV data, I need to keep only the values where z is in the range of the PIV data
wv_cfd_reduced = wv_cfd[(wv_cfd[:,0] >= wv_xmin) & (wv_cfd[:,0] <= wv_xmax) & (wv_cfd[:,1] >= wv_ymin) & (wv_cfd[:,1] <= wv_ymax)] #keep only the values where x and y are in the range of the PIV data
#TODO: FIx this
#cfd plot 
plt.figure(figsize=(12,8))
plt.title('Wake View CFD Velocity Data')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.scatter(wv_cfd_reduced[:,0], wv_cfd_reduced[:,1], c=wv_cfd_reduced[:,3], cmap='jet')
plt.colorbar(label='u (m/s)')
plt.show()

# %%
