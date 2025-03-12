#Midterm_PINN.py
#Jacob Child
#%% Packages
import numpy as np
import torch 
from torch import nn
import torch.optim as optim
from scipy.stats import qmc
from matplotlib import pyplot as plt
# %% Network
class MyPINN(nn.Module):
    def __init__(self, hidden_layers, hidden_width):
        super(MyPINN, self).__init__()
        
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
    
    
def ODE(model, collptsf): # collocation points (ncol, 1)
    #collptsf (10000,2)
    # evaluate model
    y = model(collptsf) #(10000,1)
    # compute derivatives
    dydx = torch.autograd.grad(y, collptsf, grad_outputs = torch.ones_like(y), create_graph=True)[0] # (10000,2)
    #? Will I need to reshape?
    d2ydx2 = torch.autograd.grad(dydx,collptsf, grad_outputs=torch.ones_like(dydx), create_graph=True)[0]  # we still need the graph because the optimizer will want derivatives of this as well
    # u_xx = u_all_all[:,1].reshape(10000,1)
            
    #compute residual
    f = d2ydx2 - 2* dydx + y - torch.cos(collptsf) #(10000,1)
    return torch.mean(f**2) # mean squared loss function
    

def total_loss(model, condition_points_tensorf, collocation_points_tensorf, g1, g2):  
    #Boundary condition loss y(0) = 1 and y(5) = 0
    bcn1_loss = (model(condition_points_tensorf[0]) - 1)**2
    bcn2_loss = model(condition_points_tensorf[1])**2
    phys_loss = ODE(model, collocation_points_tensorf)
    
    return g1*(bcn1_loss + bcn2_loss) + g2*phys_loss

# %% Data Point Setup, this is only a 1D problem
# Set up the boundary conditions
bc_pts = torch.tensor([[0.0], [5.0]], dtype = torch.float64, requires_grad=True)
# Set up the collocation points
ncol = 1000
sampler = qmc.LatinHypercube(d=1)
col_pts = sampler.random(ncol)
#scale the collocation points to between x = 0 and x = 5
col_pts = qmc.scale(col_pts, 0,5) #(1000,1)
#convert to tensor
col_pts = torch.tensor(col_pts, dtype = torch.float64, requires_grad=True)


# %% Training
hlayers = 2
hwidth = 20
model = MyPINN(hlayers, hwidth)
model.double()
optimizer = torch.optim.LBFGS(model.parameters(),line_search_fn="strong_wolfe", lr=1., max_iter=100) #what I used in the homework

def closure():
    optimizer.zero_grad() # zero the gradients from the last step
    train_loss = total_loss(model, bc_pts, col_pts, 1., 1.) 
    train_loss.backward()
    return train_loss

train_losses = []
epochs = 10

for e in range(epochs): 
    train_loss = optimizer.step(closure)
    train_losses.append(train_loss.item())
    if (e + 1) % 1 == 0:
        print(f"Epoch [{e + 1}/{epochs}], Loss: {train_loss.item():.4f}")

# plot so I can evaluate , my final loss was 0.000021337
plt.figure() 
plt.plot(range(epochs), train_losses, label='Train Loss')
# plt.plot(range(epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% Test and plot
model.eval()
xtest = torch.linspace(0,5,100, dtype=torch.float64).reshape(100,1)
ytest = model(xtest)

#plot
plt.figure()
plt.plot(xtest.detach().numpy(), ytest.detach().numpy(), label='Model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Prediction for: d2ydx2−2dydx+y-cos(x)=0')
plt.legend()
plt.show()


# %%
