import torch
from torch import nn

# From pytorch docs 

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__() #only called the first time
        
        self.network = nn.Sequential(
            # These are hyper parameters, number of layers and number of nodes in each layer
            nn.Linear(inputs, 32), # nn.Linear(input, output), this is the connection from the input layer to the first hidden layer
            nn.ReLU(),
            nn.Linear(32, 32), # hidden layer 1 to hidden layer 2
            nn.ReLU(),
            nn.Linear(32, outputs) #! make sure to end on linear layer, this is the connection from the last hidden layer to the output layer
        )

    def forward(self, x):
        return self.network(x)


# Define my training function 
def train(dataloader, modelf, loss_fnf, optimizerf):
    # running this all once is one epoch
    
    modelf.train()
    num_batches = len(dataloader)
    train_loss = 0
    
    for X, y in dataloader:
        # X, y = X.to(device), y.to(device)
        # print(type(X), "train func")
        # Compute prediction error
        pred = modelf(X)
        loss = loss_fnf(pred, y)

        # Backpropagation
        loss.backward() #computes derivatives
        optimizerf.step() #updates the weights or steps in a direction
        optimizerf.zero_grad() # it is doing things in place to save memory, so zeroed to start ready for next time
        train_loss += loss.item()
    
    train_loss /= num_batches
    
    print(f"Train loss: {train_loss:>8f} \n") # prints every epoch
    return train_loss
        
        
# Define my testing function
def test(dataloader, model, loss_fn):

    model.eval()
    
    num_batches = len(dataloader)
    test_loss = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            # print(type(X))
            pred = model(X)
            test_loss += loss_fn(pred, y).item() # item() gets the value of the tensor
    
    test_loss /= num_batches # average loss per batch, the /= is the same as test_loss = test_loss / num_batches
    print(f"Test loss: {test_loss:>8f} \n") # prints every epoch
    return test_loss



if __name__ == 'main': #look into this line and why it is important for scaling
    # Get cpu, gpu or mps device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = NeuralNetwork(2, 1) #initialization
    loss_fn = nn.MSELoss() #loss function: mean squared error
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # optimizer: Adam, learning rate: 1e-3
    # yhat = model(x) # this calls forward