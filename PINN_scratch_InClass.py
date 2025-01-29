#HW3_PINN.py
# template basically written by Dr. Ning from PINN_scratch_InClass.py
# Damped harmonic oscillator (Based on tutorial from Ben Moseley, 2022)
#eqn: 

# %% import needed things 
import numpy as np
import torch 
from torch import nn
import torch.optim as optim

#%% Create a fully connected network (perceptron)
# 1 input(t), 1 output(y), 4 hidden layers with width 32 (make these changeable variables), activation function = tanh (for 2nd derivatives, 2nd deriv of ReLu is 0)

class MLP(nn.Module):
    def __init__(self, hidden_layers, hidden_width):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(1,hidden_width))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Tanh()) 
            
        layers.append(nn.Linear(hidden_width,1))
        
        self.network = nn.Sequential(*layers)
            
    def forward(self, t):
        return self.network(t)
    
    
def residual(model, t): # collocation points (ncol, 1)
    m, mu, k = 1., 4., 400.
    # evaluate model
    y = model(t)
    
    # compute derivatives
    dydt = torch.autograd.grad(y, t, grad_outputs = torch.ones_like(y), create_graph=True)[0] #outputs, inputs, ie derivative of outputs with respect to inputs, can give tuple of inputs
    d2ydt2 = torch.autograd.grad(dydt,t, grad_outputs=torch.ones_like(y), create_graph=True)[0]  # we still need the graph because the optimizer will want derivatives of this as well
            
    #compute residual
    residual = m * d2ydt2 + mu * dydt + k*y
    return torch.mean(residual**2) # mean squared loss function

def boundary(model,tbc): #boundary conditions (nbc_locations,1), y(0) = 1, yprime(0) = 0
    #evaluate model
    ybc = model(tbc)
    
    #compute derivatives
    dydt = torch.autograd.grad(ybc, tbc, grad_outputs=torch.ones_like(ybc), create_graph = True)[0]
    bc1 = ybc - 1. # y(0) = 1, this is yhat_i - y_i
    bc2 = dydt - 0. #y'(0) = 0
    
    return torch.mean(bc1**2), torch.mean(bc2**2)
    

def datapoints(ncollocation): #will be random for the hw
    tbc = torch.zeros(1,1, requires_grad=True)
    tcollocation = torch.linspace(0,1,ncollocation, requires_grad=True).reshape(ncollocation,1)
    return tcollocation

def exact(t, params):
    m, mu, k = params

    delta = mu / (2*m)
    omega0 = np.sqrt(k / m)

    omega = np.sqrt(omega0**2 - delta**2)
    phi = np.arctan(-delta/omega)
    A = 1/(2*np.cos(phi))
    u = torch.exp(-delta*t)*2*A*torch.cos(phi + omega*t)
    return u

def total_loss(model, t_bc, t_col, g1, g2, g3):
    t_loss, u_loss = boundary(model, t_bc)
    phys_loss = residual(model, t_col)
    # to get an idea of magnitudes
    print("t_loss mean: ", t_loss, "u_loss: ", u_loss, "phys_loss: ", phys_loss)
    return g1*t_loss + g2*u_loss + g3*phys_loss 
    
    
    
# %% initialize data points
# data: one bc at t = 0 (ndata, numfeatures)
t_bc = torch.tensor([[0.0]], requires_grad = True) # time boundary condition points (0)
# physics: generate 50 collocation points from t= 0 to t=1
t_col = datapoints(50) #time collocation points

hlayers = 4
hwidth = 32
model = MLP(hlayers,hwidth)
# print(model.network[2].weight.shape)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# givens 
m = 1.
mu = 4. 
k = 400.
# %% training loop 
epochs = 100 # number of epochs org = 1000
train_losses = []
test_losses = []
for e in range(epochs): 
    # print(f"Epoch {e+1}\n-------------------------------")
    optimizer.zero_grad() #? what is this and why is it here?
    train_loss = total_loss(model, t_bc, t_col, 0, 000, 1)
    train_loss.backward()
    optimizer.step()
    # test_loss = test(test_dataloader, model, loss_fn)
    train_losses.append(train_loss.item())
    # test_losses.append(test_loss)
    
# plot so I can evaluate 
plt.figure() 
plt.plot(range(epochs), train_losses, label='Train Loss')
# plt.plot(range(epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %% Test 
# Evaluate the model on a finer grid
t_test = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True)
u_test = model(t_test).detach().numpy()
u_act = exact(t_test,np.array([m, mu, k]))

# Plot the results
import matplotlib.pyplot as plt

plt.plot(t_test.detach().numpy(), u_test, label='Predicted')
plt.plot(t_test.detach().numpy(), u_act.detach().numpy())
plt.scatter(t_col.detach().numpy(),np.zeros(t_col.shape))
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.show()

# %%
