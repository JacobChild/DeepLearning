#NODE_wEncoder.py
#Jacob Child
#Feb 18th, 2025

#%% load needed packages etc
from pinodedata import *
from torchdiffeq import odeint as odeint # this is the recommended ode solver for backprop reasons, it takes a function that is an nn.module, ie odeint(func, y0, t) where func is the neural net or nn.module
import torch 
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

#%% Create My network and functions etc
class MyBigNetwork(nn.Module):
    def __init__(self, nin, nmin, hidden_layers):
        super(MyBigNetwork, self).__init__()
        
        #make the networks
        self.encoder_phi_network = self.layers_maker(nin, nmin, hidden_layers)
        self.decoder_psi_network = self.layers_maker(nmin, nin, hidden_layers)
        self.ode_h_network = self.layers_maker(nmin, nmin, int(hidden_layers/2)) 
        
        
    def layers_maker(self, ninf, noutf, hidden_layersf):
        layers = []
                
        if ninf == noutf:
            for _ in range(hidden_layersf - 1):
                layers.append(nn.Linear(ninf,ninf))
                layers.append(nn.SiLU())
                
            layers.append(nn.Linear(ninf, noutf))
        else:
            #decrease or increase layer widths from ninf to noutf in the number layers given hidden_layersf
            widths = np.linspace(ninf, noutf, hidden_layersf).astype(int)
            
            for i in range(hidden_layersf - 1):
                layers.append(nn.Linear(widths[i], widths[i+1]))
                layers.append(nn.SiLU())
                
            layers.append(nn.Linear(widths[-1], widths[-1]))
            
        return nn.Sequential(*layers)
        
        
    def ode_func(self, t, y):
        return self.ode_h_network(y) #return dy/dt
    
    # def full_ode_func(self, t, y):
        return odeint(self.ode_func, y, t)

    def forward(self, xf, tsteps):
        #Pseudo code:
        # x[nbatches, ntimes, nxs]
        # z0 = encode(x[:, 0, :])
        # zhat = odeint(odefunc, z0)
        # xhat = decode(zhat)
        self.z0 = self.encoder_phi_network(xf) #[test points x nmin]
        # print('here1')
        self.zhat = odeint(self.ode_func, self.z0, tsteps).permute(1,0,2) #! changes from [11, 100, 2] to [100, 11, 2]
        self.xhat = self.decoder_psi_network(self.zhat)
        return self.xhat
    
def my_loss_func(modelf, xf, x_colf, f_colf, t_trainf, lossfnf, ef):
    #data losses
    #data loss mse(xhat,x)
    xhat = model.forward(xf[:,0,:], t_trainf)
    data_loss = lossfnf(xhat, xf)
    #data reconstruction loss mse(x, decode(encode(x)))
    data_recon_loss = lossfnf(xf, modelf.decoder_psi_network(modelf.encoder_phi_network(xf)))
    
    #physics collocation losses
    #actual physics loss mse(zdot_col, dzdt)
    z_col = modelf.encoder_phi_network(x_colf)
    zdot_col = modelf.ode_func(0.0, z_col) #? is this correct, it goes to the ode_h_network
    #calculate dzdt    
    dzdx = torch.zeros(x_colf.shape[0],n_min,x_colf.shape[1], dtype=torch.float64) #intialize
    dzdx[:, 0, :] = torch.autograd.grad(z_col[:,0], x_colf, grad_outputs=torch.ones_like(z_col[:,0]), create_graph = True)[0] #one state at a time, but you can do all batches
    dzdx[:, 1, :] = torch.autograd.grad(z_col[:,1], x_colf, grad_outputs=torch.ones_like(z_col[:,1]), create_graph = True)[0] #one state at a time, but you can do all batches

    dzdt = torch.bmm(dzdx, f_colf.unsqueeze(2)).squeeze() #batch matrix multiply, f is batch, by x, so, .unsqueeze to make it batch by x by 1, and dzdx is batch by z by x, output is batch by z by 1, I want batch by z, so .squeeze()
    col_phys_loss = lossfnf(zdot_col, dzdt)
    #reconstruction2 loss mse(x_col, decode(encode(x_col)))
    col_recon_loss = lossfnf(x_colf, modelf.decoder_psi_network(z_col))
    if (ef+1)%1000 == 0:
        print('data_loss', data_loss)
        print('data_recon_loss', data_recon_loss)
        print('col_phys_loss', col_phys_loss)
        print('col_recon_loss', col_recon_loss)
    
    return data_loss + data_recon_loss + col_phys_loss + col_recon_loss
    

# training function and loss function
def train(x_dataf, x_colf, f_colf, t_trainf, model, optimizer, lossfn, ef):
    model.train()
    optimizer.zero_grad()
    loss = my_loss_func(model, x_dataf, x_colf, f_colf, t_trainf, lossfn, ef)
    loss.backward()
    # # Check gradients
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f'Gradient for {name}: {param.grad.norm().item()}')
    
    
    optimizer.step()
    
    return loss.item()


#%% Create Data (from Dr. Ning's code)
# discretization in time for training and test data.  These don't need to be changed.
nt_train = 11
nt_test = 21
t_train = np.linspace(0.0, 1.0, nt_train)
t_test = np.linspace(0.0, 1.0, nt_test)

# number of training pts, testing pts, and collocation pts.
# You will need more training pts and collocation pts eventually (testing pts can remain as is).
ntrain = 600
ntest = 100
ncol = 1000
Xtrain, Xtest, Xcol, fcol, Amap = getdata(ntrain, ntest, ncol, t_train, t_test)
#convert to tensors
Xtrain_tensor = torch.tensor(Xtrain, dtype = torch.float64)
Xtest_tensor = torch.tensor(Xtest, dtype = torch.float64)
Xcol_tensor = torch.tensor(Xcol, dtype = torch.float64, requires_grad=True)
fcol_tensor = torch.tensor(fcol, dtype = torch.float64, requires_grad=True)
t_train_tensor = torch.tensor(t_train, dtype = torch.float64)
t_test_tensor = torch.tensor(t_test, dtype = torch.float64)

# Xtrain is ntrain x nt_train x nx
# Xtest is ntest x nt_test x nx
# Xcol is ncol x nx
# fcol is ncol x nx and represents f(Xcol)
# Amap is only needed for final plot (see function below)
#initialize the model 
hlayers = 5
num_inputs = Xtrain.shape[2]
n_min = 2
model = MyBigNetwork(num_inputs, n_min, hlayers)
model.double()
# print(model.network[2].weight.shape)
# optimizer = torch.optim.Adam(model.parameters(), lr = .1)
optimizer = optim.Adam(model.parameters(), lr = .01)
my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.5)

# training loop
epochs = 5000 #at 4500 I had 0.106 for last test loss
train_losses = []
test_losses = []
plotter_val = 500

for e in range(epochs): #range(epochs[i]):
        train_losses.append(train(Xtrain_tensor, Xcol_tensor, fcol_tensor, t_train_tensor, model, optimizer, nn.MSELoss(),e))
        my_lr_scheduler.step()
        #every 100 epochs plot the training loss and compute and plot the testing loss
        if (e+1)%plotter_val == 0:
            print('epochs done: ', e+1, '/', epochs)
            model.eval()
            Xhat_test = model.forward(Xtest_tensor[:, 0, :], t_test_tensor)
            test_loss = nn.MSELoss()(Xhat_test, Xtest_tensor)
            test_losses.append(test_loss.item())
            plt.figure()
            plt.semilogy(train_losses, label='Training Loss')
            plt.semilogy(range(plotter_val - 1, e + 1, plotter_val), test_losses, label='Testing Loss')
            plt.title('Training and Testing Losses')
            plt.xlabel('Epochs')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.show()
            plt.pause(0.001) 


# evaluate
model.eval()
Xhat = model.forward(Xtest_tensor[:, 0, :], t_test_tensor)
# test_loss = nn.MSELoss()(Xhat, Xtest_tensor)
# test_losses.append(test_loss.item())
# once you have a prediction for Xhat(t) (ntest x nt_test x nx)
# this will use this specific projection to Z, to create a plot
# like the bottom right corner of Fig 3
Zhat = true_encoder(Xhat.detach().numpy(), Amap)
Zhat_true = true_encoder(Xtest, Amap)

#%% plotting
#plot losses 
plt.figure()
plt.semilogy(train_losses, label = 'Training Loss')
plt.semilogy(range(500,5500,500),test_losses, label = 'Testing Loss') 
plt.title('Training and Testing Losses')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

plt.figure()
add_true_label = True
add_predicted_label = True

for i in range(ntest):
    plt.plot(Zhat_true[i, 0, 0], Zhat_true[i, 0, 1], "ro", alpha=0.5)
    if add_true_label:
        plt.plot(Zhat_true[i, :, 0], Zhat_true[i, :, 1], "r", alpha=0.5, label='True Zhat')
        add_true_label = False
    else:
        plt.plot(Zhat_true[i, :, 0], Zhat_true[i, :, 1], "r", alpha=0.5)
    
    plt.plot(Zhat[i, 0, 0], Zhat[i, 0, 1], "ko")
    if add_predicted_label:
        plt.plot(Zhat[i, :, 0], Zhat[i, :, 1], "k", label='Predicted Zhat')
        add_predicted_label = False
    else:
        plt.plot(Zhat[i, :, 0], Zhat[i, :, 1], "k")
    
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1, 1])

plt.legend()
plt.title('Learned Dynamics')
plt.show()

# %%
