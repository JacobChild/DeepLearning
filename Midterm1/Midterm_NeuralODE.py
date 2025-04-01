#Midterm_NeuralODE.py
#Jacob Child
#%% Packages 
import numpy as np
from torchdiffeq import odeint as odeint # this is the recommended ode solver for backprop reasons, it takes a function that is an nn.module, ie odeint(func, y0, t) where func is the neural net or nn.module
import torch 
from torch import nn
import torch.optim as optim
from matplotlib import pyplot as plt
# %% 
location1_file1 = 'node_data.csv'
location2_file1 = 'Midterm1/node_data.csv'
try: #this will run in an interactive session
    data = np.loadtxt(location1_file1) #100x6, time by 5 trajectories of position x(t) initial velocity is 0
except FileNotFoundError: #this will run if it is ran as a script
    data = np.loadtxt(location2_file1)

# Problem overview: my neural net will be dv/dt, the data I have is x(t). v = dx/dt and a = dv/dt, so I think I will use odeint twice to go from my dv/dt to x(t) and then compare that to the data I have


# %% Neural Net
class ODEFunc(nn.Module):
    def __init__(self, nin, nout):
        super(ODEFunc, self).__init__()
        
        layers = []
        layers.append(nn.Linear(nin,32))
        layers.append(nn.SiLU()) #? I may need to change the activation function
        layers.append(nn.Linear(32,32))
        layers.append(nn.SiLU()) 
        layers.append(nn.Linear(32,nout))
        # layers.append(nn.SiLU()) 
        
        # layers.append(nn.Linear(32,nout))
        
        self.network = nn.Sequential(*layers)
        
    def ode_func(self, t, state):
        x, v = state[:, 0], state[:, 1]  # Split state into x and v
        dvdt = self.network(torch.cat([x, v]))  # NN predicts acceleration
        dxdt = v  # Velocity is just dx/dt
        return torch.stack([dxdt, dvdt], dim=1)  # Return [dx/dt, dv/dt]

    def forward(self, y0, tsteps):
        #I think I will just double odeint, ie dvdt -> v -> x
        xhat = odeint(self.ode_func, y0, tsteps) #? I don't know if this works
        return xhat 

def train(xtrain, ttrain, model, optimizer, lossfn):
    
    model.train()
    optimizer.zero_grad()
    state0 = torch.stack([xtrain[0,:], torch.zeros(xtrain.shape[1])], dim = 1)
    xhat = model(state0,ttrain)
    loss = lossfn(xhat[:,:,0],xtrain)
    loss.backward()
    optimizer.step()
    
    return loss.item()

# %% data setup
#convert to tensors and split
data = torch.tensor(data, dtype=torch.float64)
x_train = data[:80,1:]
t_train = data[:80,0]
x_test = data[80:,1:]
t_test = data[80:,0]

#why not just train over all
x_train = data[:,1:]
t_train = data[:,0]

#I want to give my model t = 0 and 5 initial xs and then have it predict the rest of the x vals 
# %% setup model
num_inputs = x_train.shape[1] * 2 #x and v
num_outputs = x_train.shape[1] #just x
model = ODEFunc(num_inputs, num_outputs)
model.double()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)
# %% training loop 
epochs = 50 
train_indexes = np.array([25,50,75,100])
lossfn = nn.MSELoss()
losses = []
for i in range(4):
    for e in range(epochs):
        model.train()
        losses.append(train(x_train[:train_indexes[i],:], t_train[:train_indexes[i]], model, optimizer, lossfn))   


#%% plot losses 
plt.semilogy(losses, label='Train Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

#plot a training trajectory vs actual trajectory 
model.eval()
state0_train = torch.stack([x_train[0,:], torch.zeros(x_train.shape[1])], dim = 1)
xhat = model(state0_train,t_train) # (100 x 5 x 2)
plt.figure()
plt.plot(t_train, x_train[:,:], label='Actual')
plt.plot(t_train, xhat.detach().numpy()[:,:, 0], label='Predicted')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('Training Trajectory')
plt.show() 

# %% Evaluate the model (just one trajectory), at x = 1 and v = 0 
t_eval = torch.linspace(0,10,100, dtype=torch.float64)
state_test0 = torch.stack([torch.ones(5), torch.zeros(5)], dim = 1).double()
xhat_test = model(state_test0, t_eval)
plt.figure()
plt.plot(t_eval, xhat_test.detach().numpy()[:,0,1], label='Predicted')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Predicted Velocity Trajectory')
plt.show()

# %%
