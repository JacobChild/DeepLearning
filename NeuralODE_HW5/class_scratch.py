#class_scratch.py
#data = t, y1, y2

#%% Packages
import numpy as np
from torchdiffeq import odeint as odeint # this odeint_adjoint is the recommended ode solver for backprop reasons, it takes a function that is an nn.module, ie odeint(func, y0, t) where func is the neural net or nn.module. I couldn't get it to work
import torch 
from torch import nn
import torch.optim as optim
from matplotlib import pyplot as plt
# %% Import and plot the data to get an idea of it
fileloc1 = '../DataSets/NeuralODE_class_data.txt'
fileloc2 = 'DataSets/NeuralODE_class_data.txt'
try: 
    data = np.loadtxt(fileloc1)
except FileNotFoundError:
    data = np.loadtxt(fileloc2)
    
plt.plot(data[:,0],data[:,1], label = 'y1')
plt.plot(data[:,0],data[:,2], label = 'y2')
plt.legend()
plt.xlabel('time')

t = data[:,0]
y = data[:, 1:]

# %% Set up NN
class ODEFunc(nn.Module):
    def __init__(self, nin, hidden_layers, hidden_width, tsteps):
        super(ODEFunc, self).__init__()
        
        self.tsteps = tsteps
        layers = []
        layers.append(nn.Linear(nin,hidden_width))
        layers.append(nn.SiLU()) #? I may need to change the activation function
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.SiLU()) 
            
        layers.append(nn.Linear(hidden_width,nin))
        
        self.network = nn.Sequential(*layers)
        
    def ode_func(self, t, y):
        return self.network(y) #return dy/dt

    def forward(self, y0):
        yhat = odeint(self.ode_func, y0, self.tsteps)
        return yhat

def train(ytrain, model, optimizer, lossfn):
    
    model.train()
    optimizer.zero_grad()
    yhat = model(ytrain[0,:]) #? do I need to unsqueeze? ANS: it works without, what does it do?
    loss = lossfn(yhat,ytrain)
    loss.backward()
    optimizer.step()
    
    return loss.item()

# %% setup data 
t_train = torch.tensor(t, dtype=torch.float64)
y_train = torch.tensor(y, dtype=torch.float64)
#initialize the model
hlayers = 3
hwidth = 20
num_inputs = 2
model = ODEFunc(num_inputs, hlayers, hwidth, t_train)
model.double()
# print(model.network[2].weight.shape)
# optimizer = torch.optim.Adam(model.parameters(), lr = .1)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)


# %%
epochs = 10
lossfn = nn.MSELoss()
losses = [] 

for e in range(epochs):
    #train
    losses.append(train(y_train, model, optimizer, lossfn))
    
    if (e+1) % 50 == 0: #evaluate the model
        model.eval()
        with torch.no_grad():
            yhat = model(y_train[0,:])
            
            plt.figure()
            plt.plot(t, y[:,0], 'r')
            plt.plot(t, y[:,1], 'b')
            plt.plot(t, yhat.detach().numpy()[:,0], 'r--')
            plt.plot(t, yhat.detach().numpy()[:,1], 'b--')
            
    
    
    
        

# %%
