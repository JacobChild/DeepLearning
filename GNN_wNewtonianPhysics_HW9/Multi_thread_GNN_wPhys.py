#GNN_wPhys.py
#Jacob Child
#March 20, 2025
#%% Needed packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from scipy.integrate import solve_ivp
torch.set_num_threads(6)

# %% setup data
def data_setup(): # From Dr. Ning

    # ------------ import data ------------
    # 1000 trajectory sets with 100 time steps split 75/25 for training/testing
    loc1 = 'spring_data.npz'
    loc2 = 'GNN_wNewtonianPhysics_HW9/spring_data.npz'
    try:
        data = np.load(loc1)
    except FileNotFoundError:
        data = np.load(loc2)
    # X_train (75000, 4, 5)  data points, 4 particles, 5 states: x, y, Vx, Vy, m (positions, velocities, mass)
    data_amt = 20000
    X_train = torch.tensor(data['X_train'][0:data_amt], dtype=torch.float32)
    # y_train (75000, 4, 2)  data points, 4 particles, 2 states: ax, ay (accelerations)
    y_train = torch.tensor(data['y_train'][0:data_amt], dtype=torch.float32)
    # X_train (25000, 4, 5)  data points, 4 particles, 5 states
    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    # y_test (25000, 4, 5)  data points, 4 particles, 5 states
    y_test = torch.tensor(data['y_test'], dtype=torch.float32)
    # 100 time steps (not really needed)
    times = torch.tensor(data['times'], dtype=torch.float32)

    # ------- Save a few trajectories for plotting -------
    # the data points are currently ordered in time (for each separate trajectory)
    # so I'm going to save one set before shuffling the data.
    # this will make it easier to check how well I'm predicting the trajectories

    nt = len(times)
    train_traj = X_train[:nt, :, :]
    test_traj = X_test[:nt, :, :]

    # You can comment this out, just showing you how do this
    # for when you'll want to compare later.
    plt.figure()
    for j in range(4):  # plot all 4 particles
        plt.plot(train_traj[:, j, 0], train_traj[:, j, 1])
    plt.xlabel('x position')
    plt.ylabel('y position')

    # plotting one set of testing trajectories
    plt.figure()
    for j in range(4):
        plt.plot(test_traj[:, j, 0], test_traj[:, j, 1])
    plt.xlabel('x position')
    plt.ylabel('y position')

    plt.show()

    # ------ edge index ------
    # this just defines how the nodes (particles) are connected
    # in this case each of the 4 particles interacts with every other particle
    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
       [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
    ], dtype=torch.long)

    # -------- Further data prep ----------
    # - especially while developing (and maybe the whole time) you will want to extract
    #   just a subset of data points
    # - when you put the data into a DataLoader, you'll want to shuffle the data
    #   so that you'll pulling from different trajectories
    # - note that Data and DataLoader in torch_geometric are a bit different
    
    #make the train/test_loader 
    n = 4 #number of particles
    batch = int(64 * (4/n)**2)
    train_loader = DataLoader([Data(x=X_train[i], y=y_train[i], edge_index=edge_index) for i in range(len(X_train))], batch_size=batch, shuffle=True, num_workers=4)
    
    test_loader = DataLoader([Data(x=X_test[i], y=y_test[i], edge_index=edge_index) for i in range(len(X_test))], batch_size=batch, shuffle=True, num_workers=4)

    return train_loader, test_loader, train_traj, test_traj

# %% Data Wrangling
if __name__ == "__main__":
    training_loader, testing_loader, train_traj, test_traj = data_setup() #train_traj.shape = 100,4,5 or 100 snapshots, 4 particles, 5 states (x,y,Vx,Vy,m)
    #explore the loaders a little 
    for batch in training_loader:
        print(batch.x.shape) #shape is [batch_size, 4, 5] or [batch_size, n_particles, n_features]
        break
    # %% Needed classes and functions: model, loss
    class GN(MessagePassing):
        def __init__(self, n_f, msg_dim, ndim, hidden = 300, aggr='add'):
            super(GN, self).__init__(aggr=aggr) # add aggregation
            self.msg_func = nn.Sequential(
                nn.Linear(n_f*2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, msg_dim)
            )
            
            self.node_func = nn.Sequential(
                nn.Linear(n_f+msg_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, ndim)
            )
            
        def forward(self, x, edge_index):
            # x is [n_nodes, n_features]
            # edge_index is [2, n_edges] (#TODO: AI recommended, check shape)
            # x=x ? why is this needed?
            return self.propagate(edge_index, size = (x.size(0), x.size(0)), x=x)
        
        def message(self, x_i, x_j):
            # x_i/x_j are [n_edges, n_features]
            tmp = torch.cat([x_i, x_j], dim=1) #shape is [n_edges, 2*n_features] #TODO: AI recommended, check shape
            return self.msg_func(tmp)
        
        
        def update(self, aggr_out, x=None):
            # aggr_out is [n_nodes, msg_dim]
            tmp = torch.cat([x, aggr_out], dim=1)
            return self.node_func(tmp)
        
    def Losser(modelf, data):
        x, edge_index, y = data.x, data.edge_index, data.y
        # x.shape is [batch size, n_features (x,y,vx,vy,m)]
        # edge_index.shape is [2, n_edges?= batches * 3]
        # y.shape is [batch size, n_features (ax,ay)]
        y_hat = modelf(x, edge_index) #yhat is [batch size, n_features (ax,ay)]
        # L1 loss function
        loss = torch.mean(torch.abs(y_hat-y))
        return loss
    # %% Setup for training etc 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GN(5, 2, 2, hidden=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # %% Training loop 
    model.train()
    training_losses = []
    for epoch in range(30):
        epoch_loss = 0  # Initialize epoch loss
        batch_count = 0  # Track the number of batches
        for batch in training_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = Losser(model, batch)
            epoch_loss += loss.item()  # Accumulate batch loss
            batch_count += 1
            loss.backward()
            optimizer.step()
        # Compute average loss for the epoch
        avg_epoch_loss = epoch_loss / batch_count
        training_losses.append(avg_epoch_loss)  # Append average loss for the epoch
        print(f'Epoch {epoch}, Loss {avg_epoch_loss}')

    # Plot the training losses
    plt.plot(training_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.show()
    # %% ODE stuff to move forward
    # save my trained model 
    torch.save(model, 'model_80_epochs')
    model.eval()
    def ode_func(t,s): #double check this is the correct order
        #convert s to tensor
        s = torch.tensor(s, dtype=torch.float32).to(device).view(4,5)
        accels = model(s, edge_index)
        # Create a new state array
        new_s = np.zeros_like(s.cpu().numpy())  # Convert tensor to numpy for compatibility with solve_ivp
        # Update positions with velocities
        new_s[:, 0:2] = s[:, 2:4].cpu().detach().numpy()  # Velocities go into position slots
        # Update velocities with accelerations
        new_s[:, 2:4] = accels.cpu().detach().numpy()  # Accelerations go into velocity slots
        return new_s.flatten()

    #initial conditions
    edge_index = torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
        ], dtype=torch.long)
    x0 = train_traj[0,:,:]
    times = np.linspace(0, 5, 100)
    #solve the ode
    sol = solve_ivp(ode_func, (times[0], times[-1]), x0.flatten(), t_eval=times)
    my_trajs = sol.y.reshape(4,5,len(times))

    #plot the trajectories
    plt.figure()
    for j in range(4):
        plt.plot(my_trajs[j,0,:], my_trajs[j,1,:])
        plt.plot(train_traj[:, j, 0], train_traj[:, j, 1], '--')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('Training Predicted Trajectories')

    #plot the testing trajectories
    x0_test = test_traj[0,:,:]
    sol_test = solve_ivp(ode_func, (times[0], times[-1]), x0_test.flatten(), t_eval=times)
    my_trajs_test = sol_test.y.reshape(4,5,len(times))
    plt.figure()
    for j in range(4):
        plt.plot(my_trajs_test[j,0,:], my_trajs_test[j,1,:])
        plt.plot(test_traj[:, j, 0], test_traj[:, j, 1], '--')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('Testing Predicted Trajectories')
    plt.show()

    # %%
