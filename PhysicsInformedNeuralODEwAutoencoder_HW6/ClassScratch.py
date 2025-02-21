import torch

nb = 4 #number of batches
nx = 3 #number of x points in high dimensional space
nz = 2 # number of points in low dimensional space

x = torch.rand(nb,nx, requires_grad=True) #random x points
f = torch.zeros(nb, nz) #dynamics evaluated at high dimensional space, evaluated later

#I need some function that goes from x to z, a neural net encoder in my case
A = torch.rand(nz,nx)
def func(x):
    z = torch.zeros(nb, nz) #computed later,
    for i in range(nb):
        z[i, :] = (A @ x[i,:])**2
        return z
    
z = func(x)
dzdx = torch.zeros(nb, nz, nx) #intialize
dzdx[:, 0, :] = torch.autograd.grad(z[:,0], x, grad_outputs=torch.ones_like(z[:,0]), create_graph = True) #one state at a time, but you can do all batches
dzdx[:, 1, :] = torch.autograd.grad(z[:,0], x, grad_outputs=torch.ones_like(z[:,0]), create_graph = True) #one state at a time, but you can do all batches

dzdt = torch.bmm(dzdx, f.unsqueeze(2)).squeeze() #batch matrix multiply, f is batch, by x, so, .unsqueeze to make it batch by x by 1, and dzdx is batch by z by x, output is batch by z by 1, I want batch by z, so .squeeze()