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
    def __init__(self, nin, hidden_layers, hidden_width):
        super(ODEFunc, self).__init__()
        
        layers = []
        layers.append(nn.Linear(nin,hidden_width))
        layers.append(nn.SiLU()) #? I may need to change the activation function
        # layers.append(nn.Linear(hidden_width,32))
        # layers.append(nn.SiLU()) 
        # layers.append(nn.Linear(32,nin))
        # layers.append(nn.SiLU()) 
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.SiLU()) 
            
        layers.append(nn.Linear(hidden_width,nin))
        
        self.network = nn.Sequential(*layers)
        
    def dvdt_func(self, t, y):
        return self.network(y) #return dv/dt
    
    def veloc_func(self, t, y):
        #I'm just going to say V = a*t and v0 = 0
        self.v = self.dvdt_func(t,y) * t
        return self.v 

    def forward(self, y0, tsteps):
        #I think I will just double odeint, ie dvdt -> v -> x
        xhat = odeint(self.veloc_func, y0, tsteps) #? I don't know if this works
        return xhat 

def train(xtrain, ttrain, model, optimizer, lossfn):
    
    model.train()
    optimizer.zero_grad()
    xhat = model(xtrain[0,:],ttrain)
    loss = lossfn(xhat,xtrain)
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
hlayers = 2
hwidth = 32
num_inputs = x_train.shape[1]
model = ODEFunc(num_inputs, hlayers, hwidth)
model.double()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)
# %% training loop 
epochs = 200 
lossfn = nn.MSELoss()
losses = []
for e in range(epochs):
    losses.append(train(x_train, t_train, model, optimizer, lossfn))
    
#TODO: it would probably run better if I did the batching like in weather predictor

#%% plot losses 
plt.semilogy(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

#plot a training trajectory vs actual trajectory 
model.eval()
xhat = model(x_train[0,:],t_train)
plt.figure()
plt.plot(t_train, x_train[:,:], label='Actual')
plt.plot(t_train, xhat.detach().numpy()[:,:], label='Predicted')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('Training Trajectory')
plt.show() 


#plot a test trajectory vs actual trajectory
xhat = model(x_test[0,:],t_test)
plt.figure()
plt.plot(t_test, x_test[:,:], label='Actual')
plt.plot(t_test, xhat.detach().numpy()[:,:], label='Predicted')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('Test Trajectory')
plt.show()

# %% Evaluate the model (just one trajectory), at x = 1 and v = 0 
t_eval = torch.linspace(0,10,100, dtype=torch.float64)
# as veloc_func only takes in one time step, I will have to loop through the time steps
x0 = torch.tensor(np.array([1,1,1,1,1]), dtype=torch.float64)
vs = []
for t in t_eval:
    vs.append(model.veloc_func(t,x0).detach().numpy())
vs = np.array(vs)
plt.figure()
plt.plot(t_eval, vs[:,0], label='Predicted')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Predicted Velocity Trajectory')
plt.show()

# %%
