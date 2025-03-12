#Midterm_NeuralKoopman.py
# Jacob Child

#%% import packages
import numpy as np
import torch 
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt

# %% My Models and functions
class MyBigNetwork(nn.Module):
    def __init__(self, nin, nmid, hidden_layers):
        super(MyBigNetwork, self).__init__()
        
        #make the networks
        #encoder has nin inputs and nmid outputs with one hidden layer of width 10 
        enc_layers = [] 
        enc_layers.append(nn.Linear(nin, 10))
        enc_layers.append(nn.SiLU())
        enc_layers.append(nn.Linear(10, nmid))
        self.enc_phi_net = nn.Sequential(*enc_layers)
        #decode is reverse 
        dec_layers = []
        dec_layers.append(nn.Linear(nmid, 10))
        dec_layers.append(nn.SiLU())
        dec_layers.append(nn.Linear(10, nin))
        self.dec_psi_net = nn.Sequential(*dec_layers)
        #K I will do a single hidden layer for now
        K_layers = []
        K_layers.append(nn.Linear(nmid, nmid))
        K_layers.append(nn.SiLU())
        K_layers.append(nn.Linear(nmid, nmid))
        self.K_Layer = nn.Sequential(*K_layers)
        #now try with only one K layer
        self.K_Layer = nn.Sequential(nn.Linear(nmid, nmid))
        
    
    def K_prop(self, zk0f, tf):
        zks = []
        zks.append(zk0f)
        for _ in range(1,tf.shape[0]): #iterate over all of the time steps (I am predicting, so I stop one short)
            zks.append(self.K_Layer(zks[-1]))
            
        return torch.stack(zks, dim = 1).permute(1,0) #!this didn't need to be there on the homework?

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
    L_lin = lossfnf(enc_xks, modelf.K_prop(enc_xks[0,:], tsf))
    L_pred = lossfnf(xkf, modelf.forward(xkf[0, :], tsf))
    # additional term to prevent arbitrary scaling, the latent space z should have a mean of 0 and std of 1
    L_latent = lossfnf(torch.zeros_like(torch.mean(enc_xks,dim=1)),torch.mean(enc_xks,dim=1)) + lossfnf(torch.std(enc_xks,dim=1), torch.ones_like(torch.std(enc_xks,dim=1)))
    
    return L_recon + L_pred + L_lin + L_latent

#%% data 
nt = 30
xtrain = torch.zeros(nt, 1, dtype=torch.float64)
xtrain[0] = 0.5
for i in range(1, nt):
    xtrain[i] = 3.7*xtrain[i-1]*(1 - xtrain[i-1])

ttrain = torch.linspace(1,nt,nt)#.reshape(-1,1)
#%% Training run
model = MyBigNetwork(nin=1, nmid=3, hidden_layers=1)
model.double()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_fn = nn.MSELoss()

epochs = 15000 #30000
plotter_val = 1000 #1000
train_losses = []
test_losses = []

for e in range(epochs): #range(epochs[i]):
        model.train()
        optimizer.zero_grad()
        #? train just the encoder first
        if e < 25:
            loss = loss_fn(xtrain, model.dec_psi_net(model.enc_phi_net(xtrain)))
        else:
            loss = my_loss_func(model,xtrain, ttrain, loss_fn, e)
        
        train_losses.append(loss.detach().numpy())
        
        loss.backward()
        optimizer.step()
            
        #every 100 epochs plot the training loss and compute and plot the testing loss
        if (e+1)%plotter_val == 0:
            print('epochs done: ', e+1, '/', epochs)
            plt.figure()
            plt.semilogy(train_losses, label='Training Loss')
            plt.title('Training Losses')
            plt.xlabel('Epochs')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.show()
            plt.pause(0.001)
# %% K eigenvalues
#My final loss was at about 0.0147, but my loss plot had kind of plateaued at 6000 epochs
eigvals = torch.linalg.eigvals(model.K_Layer[0].weight)
print(eigvals)
#Correct output
# tensor([ 0.2048+0.0000j, -0.9448+0.3228j, -0.9448-0.3228j],
#        dtype=torch.complex128, grad_fn=<LinalgEigBackward0>)

# old when I had two layers
# eigvals2 = torch.linalg.eigvals(model.K_Layer[2].weight)
# print(eigvals2)
#Returns:
# tensor([-0.2675+1.0707j, -0.2675-1.0707j,  0.0679+0.0000j],
#        dtype=torch.complex128, grad_fn=<LinalgEigBackward0>)
# tensor([ 0.5510+0.7496j,  0.5510-0.7496j, -0.6976+0.0000j],
#        dtype=torch.complex128, grad_fn=<LinalgEigBackward0>)
# %%
