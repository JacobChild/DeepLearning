#HW4.py
#Jacob Child
#February 5th, 2025

#! .venv\Scripts\Activate.ps1

#%% Needed packages
import numpy as np
import torch 
from torch import nn
import torch.optim as optim
from scipy.stats import qmc
from matplotlib import pyplot as plt

#%% Create a fully connected network (perceptron)

class MLP(nn.Module):
    def __init__(self, hidden_layers, hidden_width):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(2,hidden_width))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Tanh()) 
            
        layers.append(nn.Linear(hidden_width,1))
        
        self.network = nn.Sequential(*layers)
        
        #parameters for Burgers equation
        self.params = nn.Parameter(torch.rand(2,requires_grad=True)) #lambda1 and lambda2 coefficients, should become 1 and -.01 / pi
            
    def forward(self, t):
        return self.network(t)
    
    
def Burgers(model, collptsf): # collocation points (ncol, 1)
    #collptsf (10000,2)
    # evaluate model
    u = model(collptsf) #(10000,1)
    # compute derivatives
    u_all = torch.autograd.grad(u, collptsf, grad_outputs = torch.ones_like(u), create_graph=True)[0] # (10000,2)
    # print("shape of u_all: ", u_all.shape)#outputs, inputs, ie derivative of outputs with respect to inputs, can give tuple of inputs
    u_t = u_all[:,0].reshape(10000,1)
    u_x = u_all[:,1].reshape(10000,1)
    u_all_all = torch.autograd.grad(u_x,collptsf, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]  # we still need the graph because the optimizer will want derivatives of this as well
    u_xx = u_all_all[:,1].reshape(10000,1)
            
    #compute residual
    l1 = model.params[0]
    l2 = model.params[1]
    f = u_t + l1*u*u_x + l2*u_xx #(10000,1)
    return torch.mean(f**2) # mean squared loss function

def data_loss_func(model,dataptsf): #boundary and initial conditions (nbc_locations,1)
    #evaluate model
    # dataptsf (length of data,3)
    uc = model(dataptsf[:,[0,1]]) # (length of data,1)
    u_data = dataptsf[:,2].reshape(uc.shape[0],1) # (length of data, 1)
    u_loss = u_data - uc
    # print('u_data.shape: ', u_data.shape)
    
    return torch.mean(u_loss**2)
    

def total_loss(model, data_points_tensorf, collocation_points_tensorf, g1, g2):
    
    data_loss = data_loss_func(model, data_points_tensorf)
    #print(ic_loss,'in total_loss')
    phys_loss = Burgers(model, collocation_points_tensorf)
    # to get an idea of magnitudes
    # print("g1*ic_loss mean: ", ic_loss, "g2*bcn1_loss: ", bcn1_loss, "g3*bcp1_loss: ", bcp1_loss, "g4*phys_loss: ", phys_loss)
    return g1*data_loss + g2*phys_loss
    
    
    
# %% Collocation and data points
N_collocation = 10000
#Collocation points with latin hypercube sampling
# Define the sampler
sampler = qmc.LatinHypercube(d=2)  # d=2 for two dimensions (x and t)
# Generate LHS samples
lhs_samples = sampler.random(N_collocation)
# Scale the samples to the desired range
samples = qmc.scale(lhs_samples, [-1,0], [1,1])
xcoll = samples[:,0].reshape(N_collocation,1)
tcoll = samples[:,1].reshape(N_collocation,1)
#Combine and turn into a tensor 
collocation_points = np.hstack((tcoll,xcoll))
collocation_points_tensor = torch.tensor(collocation_points, dtype=torch.float64, requires_grad=True)

# Load Data Points
fileloc = '../DataSets/HW4Data/BurgersData.txt'
alternative_fileloc = 'DataSets/HW4Data/BurgersData.txt'

try:
    datapts = np.loadtxt(fileloc, dtype=np.float64)
except FileNotFoundError:
    print(f"File not found at {fileloc}. Trying alternative path...")
    datapts = np.loadtxt(alternative_fileloc, dtype=np.float64)

#swap columns, ie change from x,t,u to t,x,u so the new rows are 1,0,2
new_datapts = datapts[:, [1, 0, 2]]
datapts_tensor = torch.tensor(new_datapts, dtype=torch.float64, requires_grad=True)

#initialize the model
hlayers = 9
hwidth = 20
model = MLP(hlayers,hwidth)
model.double()
# print(model.network[2].weight.shape)
# optimizer = torch.optim.Adam(model.parameters(), lr = .1)
optimizer = torch.optim.LBFGS(model.parameters(),line_search_fn="strong_wolfe", lr=1., max_iter=200)

# %% training loop 
train_losses = []
l1_model = []
l2_model = []
test_losses = []
epochs = 15 # number of epochs org = 1000

def closure():
    optimizer.zero_grad() # zero the gradients from the last step
    train_loss = total_loss(model, datapts_tensor, collocation_points_tensor, 1., 1.3) #
    train_loss.backward()
    return train_loss

for e in range(epochs): 
    train_loss = optimizer.step(closure)
    train_losses.append(train_loss.item())
    l1_model.append(model.params[0])
    l2_model.append(model.params[1])
    if (e + 1) % 1 == 0:
        print(f"Epoch [{e + 1}/{epochs}], Loss: {train_loss.item():.4f}")
    
# plot so I can evaluate 
plt.figure() 
plt.plot(range(epochs), train_losses, label='Train Loss')
# plt.plot(range(epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('l1 value (should be about 1 +/- 0.1): ', model.params[0])
print('l2 value (should be about -.003183 +/- 0.001): ', model.params[1])

# %% Test and plot
ttest_slice = np.ones(100) * 0.75
ttest_slice = ttest_slice.reshape(100,1)
xtest_slice = np.linspace(-1.,1.,100).reshape(100,1)
#convert to tensors 
txtest_slice = np.hstack((ttest_slice, xtest_slice))
txtest_slice_tensor = torch.tensor(txtest_slice, dtype=torch.float64)
#evaluate the model
uslice = model(txtest_slice_tensor)

# Plot the results


plt.plot(xtest_slice, uslice.detach().numpy(), label='Predicted')
# plt.plot(xtest_slice, -np.sin(np.pi*xtest_slice), label = 'initial cond')
plt.xlabel('x')
plt.ylabel('u(t,x)')
plt.legend()
plt.title('t = 0.75')
plt.show()

#contour plot now

# %% contour plot (thanks to AI help)
t_values = np.linspace(0, 1, 100)
x_values = np.linspace(-1, 1, 100)
T, X = np.meshgrid(t_values, x_values)
tx_grid = np.hstack((T.reshape(-1, 1), X.reshape(-1, 1)))
tx_grid_tensor = torch.tensor(tx_grid, dtype=torch.float64)

# Evaluate the model on the grid
u_grid = model(tx_grid_tensor).detach().numpy().reshape(100, 100)

# Create the contour plot
plt.figure()
cp = plt.contourf(T, X, u_grid, levels=50, cmap='rainbow')
plt.colorbar(cp)
plt.xlabel('Time (t)')
plt.ylabel('x')
plt.title('Contour plot of u(t,x)')
plt.show()
# %%
