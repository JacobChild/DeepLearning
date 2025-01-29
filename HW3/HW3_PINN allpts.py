#HW3_PINN.py
# template basically written by Dr. Ning from PINN_scratch_InClass.py
# going from [this](https://www.sciencedirect.com/science/article/pii/S0021999118307125?ref=cra_js_challenge&fr=RR-1#se0140) paper
#eqn: from appendix A.1 
#In one space dimension, the Burger’s equation along with Dirichlet boundary conditions reads as
# ut + uux − (0.01/π)uxx = 0, x ∈ [−1, 1], t ∈ [0, 1], (A.1)
#u(0, x) = −sin(πx),
#u(t,−1) = u(t, 1) = 0.
#Let us define f (t, x) to be given by
#f := ut + uux − (0.01/π)uxx,

#! .venv\Scripts\Activate.ps1

# %% import needed things 
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
            
    def forward(self, t):
        return self.network(t)
    
    
def Burgers(model, allptstensorf): # collocation points (ncol, 1)
    #collptsf (10000,2)
    # evaluate model
    u = model(allptstensorf) #(10000,1)
    u = u[range(100,10100),:] # (10000,2)
    # compute derivatives
    u_all = torch.autograd.grad(u, allptstensorf[range(100,10100),:], grad_outputs = torch.ones_like(u), create_graph=True)[0] # (10000,2)
    # print("shape of u_all: ", u_all.shape)#outputs, inputs, ie derivative of outputs with respect to inputs, can give tuple of inputs
    u_t = u_all[:,0].reshape(10000,1)
    u_x = u_all[:,1].reshape(10000,1)
    u_all_all = torch.autograd.grad(u_x,allptstensorf[range(100,10100),:], grad_outputs=torch.ones_like(u_x), create_graph=True)[0]  # we still need the graph because the optimizer will want derivatives of this as well
    u_xx = u_all_all[:,1].reshape(10000,1)
            
    #compute residual
    f = u_t + u*u_x - (0.01/  np.pi)*u_xx #(10000,1)
    return torch.mean(f**2) # mean squared loss function

def conditions(model,allptstensorf): #boundary and initial conditions (nbc_locations,1)
    #evaluate model
    # condptsf (100,2)
    uc = model(allptstensorf) # (100,1)
    t0pts = range(0,50)
    xn1pts = range(50,75) #includes the left and excludes the right, 
    xp1pts = range(75,100)
    #Initial condition: u(0, x) = −sin(πx)
    #print(uc[t0pts].shape, "xshape: ",condptsf[t0pts,1].reshape(50,1).shape)
    
    ic_loss = uc[t0pts] + torch.sin(torch.pi * allptstensorf[t0pts,1].reshape(50,1)) # should be (50,1)
    #print(ic_loss.shape, "in conditions")
    #Boundary Condition: u(t,−1) = u(t, 1) = 0
    bcn1_loss = uc[xn1pts] # (25,1)
    bcp1_loss = uc[xp1pts] # (25,1)
    #print(bcn1_loss.shape, 'shape')
    
    return torch.mean(ic_loss**2), torch.mean(bcn1_loss**2), torch.mean(bcp1_loss**2)
    

def total_loss(model, allptstensorf, g1, g2, g3, g4):
    condition_points_tensorf = allptstensorf[range(0,100),:] #(100,2)
    collocation_points_tensorf = allptstensorf[range(100,10100),:] # (10000,2)
    ic_loss, bcn1_loss, bcp1_loss = conditions(model, allptstensorf)
    #print(ic_loss,'in total_loss')
    phys_loss = Burgers(model, allptstensorf)
    # to get an idea of magnitudes
    #print("g1*ic_loss mean: ", ic_loss.shape, "g2*bcn1_loss: ", bcn1_loss.shape, "g3*bcp1_loss: ", bcp1_loss.shape, "g4*phys_loss: ", phys_loss.shape)
    return g1*ic_loss + g2*bcn1_loss + g3*bcp1_loss + g4*phys_loss
    
    
    
# %% initialize data points
Nu = 100 #(ie for xs)
N_collocation = 10000

# xc and tc (condition) points 
xc = np.ones(Nu)
tc = np.zeros(Nu)
xc[0:50] = np.random.uniform(-1,1,50)
xc[50:76] = -1.
xc[76:] = 1.
tc[50:] = np.random.uniform(0,1,50)
xc = xc.reshape(Nu,1)
tc = tc.reshape(Nu,1)
# Combine tc and xc into a single array and convert to a tensor
condition_points = np.hstack((tc, xc))
condition_points_tensor = torch.tensor(condition_points, dtype=torch.float32, requires_grad=True)

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
collocation_points_tensor = torch.tensor(collocation_points, dtype=torch.float32, requires_grad=True)

# Combine all points collocation and condition points into one tensor similar to np.vstack
all_points_tensor = torch.cat((condition_points_tensor, collocation_points_tensor), dim=0)


#initialize the model
hlayers = 9
hwidth = 20
model = MLP(hlayers,hwidth)
# print(model.network[2].weight.shape)
optimizer = torch.optim.Adam(model.parameters(), lr = 1.0)

# %% training loop 
epochs = 20 # number of epochs org = 1000
train_losses = []
test_losses = []
for e in range(epochs): 
    optimizer.zero_grad() # zero the gradients from the last step
    train_loss = total_loss(model, all_points_tensor, 1., 0, 0, 0.) #I had 1,10,10,100000: ic, bc (neg), bc(pos), data/physics
    train_loss.backward()
    optimizer.step()
    # test_loss = test(test_dataloader, model, loss_fn)
    train_losses.append(train_loss.item())
    # test_losses.append(test_loss)
    if (e + 1) % 20 == 0:
        print(f"Epoch [{e + 1}/{epochs}], Loss: {train_loss.item():.4f}")
    
# plot so I can evaluate 
plt.figure() 
plt.plot(range(epochs), train_losses, label='Train Loss')
# plt.plot(range(epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %% Test 
ttest_slice = np.ones(100) * 0.
ttest_slice = ttest_slice.reshape(100,1)
xtest_slice = np.linspace(-1.,1.,100).reshape(100,1)
#convert to tensors 
txtest_slice = np.hstack((ttest_slice, xtest_slice))
txtest_slice_tensor = torch.tensor(txtest_slice, dtype=torch.float32)
#evaluate the model
uslice = model(txtest_slice_tensor)

# Plot the results


plt.plot(xtest_slice, uslice.detach().numpy(), label='Predicted')
plt.plot(xtest_slice, -np.sin(np.pi*xtest_slice), label = 'initial cond')
plt.xlabel('x')
plt.ylabel('u(t,x)')
plt.legend()
plt.title('initial condition test')
plt.show()

# %%
