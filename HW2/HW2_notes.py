import numpy as np
import math
import matplotlib.pyplot as plt

# -------- activation functions -------
def relu(z):
    #TODO_check
    return np.maximum(0,z)

def relu_back(xbar, z):
    #TODO_check 
    return xbar * (z > 0) #derivative of x with respect to z of the activation layer, ie derivative of the outputs of relu with respect to its inputs (z)
    

identity = lambda z: z # a lambda function returns the input as the output unchanged

identity_back = lambda xbar, z: xbar # derivative of x with respect to z in the linear layer
# -------------------------------------------


# ---------- initialization -----------
def glorot(nin, nout):
   # TODO_check
   # we want 0 mean
   np.random.seed(0) #for repeatability
   var = np.sqrt(6 / (nin + nout))
   W = np.random.normal(0, var, (nout, nin))# size nout x nin
   b = np.zeros((nout,1)) # nout x 1
#    print('weights:', W)
#    print('Biases:', b)
   return W, b
# -------------------------------------


# -------- loss functions -----------
def mse(yhat, y):
    # TODO_check
    return (1/np.size(yhat))*np.sum((yhat - y)**2)

def mse_back(yhat, y):
    #
    return 2.0 / np.size(yhat) * (yhat - y)
# -----------------------------------


# ------------- Layer ------------
class Layer:

    def __init__(self, nin, nout, activation=identity):
        # TODO: initialize and setup variables
        self.W, self.b = glorot(nin, nout)
        self.activation = activation
        
        
        if activation == relu:
            self.activation_back = relu_back
        if activation == identity:
            self.activation_back = identity_back

        # initialize cache
        self.cache = {}

    def forward(self, X, train=True):
        # TODO_check
        # Z is a matrix of size (nodes, data points) and contains the Z for each node and data point combo, essentially it has the current layer for all 313 data points saved at once, so we are processing all of them 
        Z = np.dot(self.W, X) + self.b 
        Xnew = self.activation(Z)        
        
        # save cache
        if train:
            #save cache
            # Save intermediate values needed for backpropagation
            self.cache['X'] = X
            self.cache['Z'] = Z
            self.cache['Xnew'] = Xnew

        return Xnew

    def backward(self, Xnewbar):
        # TODO_check 
        X = self.cache['X']
        Z = self.cache['Z']
        Zbar = self.activation_back(Xnewbar, Z) # derivative of the loss with respect to Z, ie the output of the linear layer
        self.Wbar = np.dot(Zbar, X.T) #derivative of the loss with respect to the weights
        self.bbar = np.sum(Zbar, axis=1, keepdims=True) #derivative of the loss with respect to the biases
        Xbar = np.dot(self.W.T, Zbar) #derivative of the loss with respect to the output of the previous layer, this becomes saved as yhat_bar, essentially the derivative of the loss with respect to the input of the current layer
        return Xbar


class Network:

    def __init__(self, layers, loss):
        # TODO: initialization_check 
        self.layers = layers 
        self.loss = loss 
        self.cache = {}

        if loss == mse:
            self.loss_back = mse_back

    def forward(self, X, y, train=True):
        # TODO_check 
        for layer in self.layers:
            X = layer.forward(X, train)
        yhat = X
        L = self.loss(yhat, y)

        # save cache
        if train:
            self.cache['y'] = y
            self.cache['yhat'] = yhat
            
        return L, yhat

    def backward(self):
        # TODO_check
        y = self.cache['y']
        yhat = self.cache['yhat']
        yhat_bar = self.loss_back(yhat, y) #derivative of the loss with respect to the output (prediction) for each data point, so size is (1,datapoints) as there is only one final node or prediction
        
        for layer in reversed(self.layers):
            # now take the derivative of the output of one layer with respect to the input of that layer (ie the output of the previous layer)
            yhat_bar = layer.backward(yhat_bar)
        



class GradientDescent:

    def __init__(self, alpha):
        # TODO_check 
        self.alpha = alpha

    def step(self, network):
        # TODO_check 
        for layer in network.layers:
            layer.W -= self.alpha * layer.Wbar # W = W - alpha*dL/dw, the weights for the new layer now equal the rates of the last layer minus the derivative of the loss with respect to the weights (ie how much that weight matters) multiplied by some learning rate.
            layer.b -= self.alpha * layer.bbar


if __name__ == '__main__':

    # ---------- data preparation ----------------
    # Initialize lists for the numeric data and the string data
    numeric_data = []

    # Read the text file
    with open('DataSets/HW1Data/auto-mpg.data', 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.strip().split()

            # Check if any of the first 8 columns contain '?'
            if '?' in columns[:8]:
                continue  # Skip this line if there's a missing value

            # Convert the first 8 columns to floats and append to numeric_data
            numeric_data.append([float(value) for value in columns[:8]])

    # Convert numeric_data to a numpy array for easier manipulation
    numeric_array = np.array(numeric_data)

    # Shuffle the numeric array and the corresponding string array
    nrows = numeric_array.shape[0]
    indices = np.arange(nrows)
    np.random.shuffle(indices)
    shuffled_numeric_array = numeric_array[indices]

    # Split into training (80%) and test (20%) sets
    split_index = int(0.8 * nrows)

    train_numeric = shuffled_numeric_array[:split_index]
    test_numeric = shuffled_numeric_array[split_index:]

    # separate inputs/outputs
    Xtrain = train_numeric[:, 1:]
    ytrain = train_numeric[:, 0]

    Xtest = test_numeric[:, 1:]
    ytest = test_numeric[:, 0]

    # normalize
    Xmean = np.mean(Xtrain, axis=0)
    Xstd = np.std(Xtrain, axis=0)
    ymean = np.mean(ytrain)
    ystd = np.std(ytrain)

    Xtrain = (Xtrain - Xmean) / Xstd
    Xtest = (Xtest - Xmean) / Xstd
    ytrain = (ytrain - ymean) / ystd
    ytest = (ytest - ymean) / ystd

    # reshape arrays (opposite order of pytorch, here we have nx x ns).
    # I found that to be more conveient with the way I did the math operations, but feel free to setup
    # however you like.
    Xtrain = Xtrain.T # 7 x 313, ie 7 features and 313 data points
    Xtest = Xtest.T
    ytrain = np.reshape(ytrain, (1, len(ytrain))) # 1 x 313, the expected outputs for the 313 data points
    ytest = np.reshape(ytest, (1, len(ytest)))

    # My Notes on shapes: rows are a data point and columns are features

    # ------------------------------------------------------------
    # TODO_check
    # The first time Layer is called it does the init section of the class, including calling glorot and initializing the weights and biases and specifying the activation function and initializes a cache, so each layer has its own sets of weights, biases, activation function, and biases. l1:3 are objects of Class layer
    l1 = Layer(7, 52, relu)
    l2 = Layer(52, 25, relu)
    l3 = Layer(25, 1, identity) #? what does identity do?

    # The first time Network is called it initializes the network and saves layers and the loss function (as well as the back function for mse), the network object also has the forward and back functions now as well.
    layers = [l1, l2, l3]
    network = Network(layers, mse)
    
    # The first time GradientDescent is called it initializes the object, saves alpha (the learning rate), and the step function
    alpha = 0.001
    optimizer = GradientDescent(alpha)

    train_losses = []
    test_losses = []
    epochs = 2000
    for i in range(epochs):
        # TODO: run train set, backprop, step
        # this calls the forward function from the network object and has it run on the training data. the network.forward func iterates through the number of layers I have and runs layer.forward()
        #layer.forward() runs my linear layer Z = WX + b, then ReLu, and along the way saves the relevant values to the cache for that respective layer
        #Network outputs the Loss and yhat (the predicted y values), but it doesn't need yhat until the final test.
        train_loss, _ = network.forward(Xtrain, ytrain, train=True)
        
        network.backward() # computes all of the derivatives for each layer, ie the derivative of the loss with respect to the inputs of a given layer. Only the Weights and biases' derivatives are saved inside of each layer function
        optimizer.step(network) #updates the weights and biases using simple gradient descent, ie based off of the gradients of the Loss with respect to the weights and biases, try to minimize Loss because of how much a given weight or bias is messing it up.

        # Do the above, but no need to cache or calculate derivatives as it is on the test data
        test_loss, _ = network.forward(Xtest, ytest, train=False)
        train_losses.append(train_loss)
        test_losses.append(test_loss)


    # --- inference ----
    _, yhat = network.forward(Xtest, ytest, train=False)

    # unnormalize
    yhat = (yhat * ystd) + ymean
    ytest = (ytest * ystd) + ymean

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.legend()


    plt.figure()
    plt.plot(ytest.T, yhat.T, "o")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.plot([10, 45], [10, 45], "--")

    print("avg error (mpg) =", np.mean(np.abs(yhat - ytest)))

    plt.show()
    # I got 2.6664mpg orginally