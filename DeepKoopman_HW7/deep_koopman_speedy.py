#deep_koopman_speedy.py
#AI recommended batching and multithreading?
#Jacob Child
#Feb 26, 2025

#%%
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# %% My Models and functions
class MyBigNetwork(nn.Module):
    def __init__(self, nin, nmid, hidden_layers):
        super(MyBigNetwork, self).__init__()
        
        #make the networks
        self.enc_phi_net = self.layers_maker(nin, nmid, hidden_layers)
        self.dec_psi_net = self.layers_maker(nmid, nin, hidden_layers)
        self.K_Layer = self.layers_maker(nmid, nmid, 1) 
        
        
    def layers_maker(self, ninf, noutf, hidden_layersf):
        layers = []
                
        if ninf == noutf:
            # for _ in range(hidden_layersf - 1):
            #     layers.append(nn.Linear(ninf,ninf))
            #     layers.append(nn.SiLU())
                
            layers.append(nn.Linear(ninf, noutf, bias=False))
        else:
            #decrease or increase layer widths from ninf to noutf in the number layers given hidden_layersf
            widths = np.linspace(ninf, noutf, hidden_layersf).astype(int)
            
            for i in range(hidden_layersf - 1):
                layers.append(nn.Linear(widths[i], widths[i+1]))
                layers.append(nn.SiLU())
                
            layers.append(nn.Linear(widths[-1], widths[-1]))
            
        return nn.Sequential(*layers)
    
    def K_prop(self, zk0f, tf):
        zks = []
        zks.append(zk0f)
        for _ in range(1,tf.shape[0]): #iterate over all of the time steps (I am predicting, so I stop one short)
            zks.append(self.K_Layer(zks[-1]))
            
        return torch.stack(zks, dim = 1)

    def forward(self, xkf, tsteps):
        #Pseudo code: encode xf, then advance it forward in time using K, then decode
        z_k = self.enc_phi_net(xkf)
        z_k1 = self.K_prop(z_k, tsteps)
        x_k1 = self.dec_psi_net(z_k1)
        return x_k1
    

def my_loss_func(modelf, xkf, tsf, lossfnf, ef):
    #L_recon = lossfnf(x1, phi^-1(phi(x1)) 
    #L_pred = 1. /Sp * sum(lossfnf(x_(m+1), phi^-1(phi(x1)))) #mse(phi(X_m), K^m*phi(x0))
    #L_lin = 1. / (T-1) * sum(lossfnf(phi(x_(m+1) - K^m * phi(x1)))) #mse(xm, psi(K^m*(phi(x0))))
    enc_xks = modelf.enc_phi_net(xkf)
    L_recon = lossfnf(xkf, modelf.dec_psi_net(enc_xks))
    L_lin = lossfnf(enc_xks, modelf.K_prop(enc_xks[:,0,:], tsf))
    L_pred = lossfnf(xkf, modelf.forward(xkf[:, 0, :], tsf))
    
    # if (ef+1)%1000 == 0:
    #     print('data_loss', data_loss)
    #     print('data_recon_loss', data_recon_loss)
    #     print('col_phys_loss', col_phys_loss)
    #     print('col_recon_loss', col_recon_loss)
    
    return L_recon + L_pred + L_lin
    
    
#%% Load Data
file_loc1 = 'kdata.txt'
file_loc2 = 'DeepKoopman_HW7/kdata.txt'
ntraj = 2148  # number of trajectories, max 2148
nt = 50  # number of time steps
ny = 7  # number of states
train_split = 1000

tvec = np.linspace(0, 350, nt)
try: #this will run in an interactive session
    Y = np.loadtxt(file_loc1).reshape(ntraj, nt, ny)
except FileNotFoundError: #this will run if it is ran as a script
    Y = np.loadtxt(file_loc2).reshape(ntraj, nt, ny)
# Y = np.loadtxt('kdata.txt').reshape(ntraj, nt, ny)
Ytrain = Y[:train_split, :, :]  # 2048 training trajectories
Ytest = Y[train_split:train_split+100, :, :]  # 100 testing trajectories, ntraj, ntime, nstates

# Convert to tensors
tvec_tensor = torch.tensor(tvec, dtype=torch.float64)
Ytrain_tensor = torch.tensor(Ytrain, dtype=torch.float64, requires_grad=True)
Ytest_tensor = torch.tensor(Ytest, dtype=torch.float64, requires_grad=False)

# Create DataLoader for batching
train_dataset = TensorDataset(Ytrain_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

#%% Training run
model = MyBigNetwork(nin=ny, nmid=15, hidden_layers=3)
model.double()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 3000
plotter_val = 500
train_losses = []
test_losses = []

for e in range(epochs): #range(epochs[i]):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        Ytrain_batch = batch[0]
        # Train just the encoder first
        if e < 25:
            loss = loss_fn(Ytrain_batch, model.dec_psi_net(model.enc_phi_net(Ytrain_batch)))
        else:
            loss = my_loss_func(model, Ytrain_batch, tvec_tensor, loss_fn, e)
        
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    # Every plotter_val epochs, plot the training loss and compute and plot the testing loss
    if (e + 1) % plotter_val == 0:
        print('epochs done: ', e + 1, '/', epochs)
        model.eval()
        with torch.no_grad():
            Zpred = model.forward(Ytest_tensor[:,0,:], tvec_tensor)
            test_loss = loss_fn(Ytest_tensor, Zpred).item()
            test_losses.append(test_loss)
        
        plt.figure()
        plt.semilogy(train_losses, label='Training Loss')
        plt.semilogy(range(plotter_val - 1, e + 1, plotter_val), test_losses, label='Testing Loss')
        plt.title('Training and Testing Losses')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.show()
        plt.pause(0.001)

# %% Plot Trajectories, just the first 3 states
model.eval()
Zpred_train = model.forward(Ytrain_tensor[:,0,:], tvec_tensor)

plt.plot(tvec, Ytrain[0,:,0], 'r--', label='True State 1')  # Dashed line for true state 1
plt.plot(tvec, Ytrain[0,:,1], 'b--', label='True State 2')  # Dashed line for true state 2
plt.plot(tvec, Ytrain[0,:,2], 'g--', label='True State 3')  # Dashed line for true state 3

plt.plot(tvec, Zpred_train.detach().numpy()[0, :, 0], 'r-', label='Predicted State 1', alpha=0.5)  # Solid line for predicted state 1
plt.plot(tvec, Zpred_train.detach().numpy()[0, :, 1], 'b-', label='Predicted State 2', alpha=0.5)  # Solid line for predicted state 2
plt.plot(tvec, Zpred_train.detach().numpy()[0, :, 2], 'g-', label='Predicted State 3', alpha=0.5)  # Solid line for predicted state 3

plt.xlabel('Time')
plt.ylabel('State Value')
plt.title('True vs Predicted States')
plt.legend()
plt.show()