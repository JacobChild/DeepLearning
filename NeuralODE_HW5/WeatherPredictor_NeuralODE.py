#NeuralODE_HW5.py
#Jacob Child
#Feb 10, 2025

#! .venv\Scripts\Activate.ps1

# %% Import packages etc
import pandas as pd
import numpy as np
from torchdiffeq import odeint as odeint # this is the recommended ode solver for backprop reasons, it takes a function that is an nn.module, ie odeint(func, y0, t) where func is the neural net or nn.module
import torch 
from torch import nn
import torch.optim as optim
from matplotlib import pyplot as plt
# %% Import Data
location1_file1 = '../DataSets/Weather_HW5/DailyDelhiClimateTrain.csv'
location1_file2 = '../DataSets/Weather_HW5/DailyDelhiClimateTest.csv'
location2_file1 = 'DataSets/Weather_HW5/DailyDelhiClimateTrain.csv'
location2_file2 = 'DataSets/Weather_HW5/DailyDelhiClimateTest.csv'

try: #this will run in an interactive session
    data_train = pd.read_csv(location1_file1)
    data_test = pd.read_csv(location1_file2)
except FileNotFoundError: #this will run if it is ran as a script
    data_train = pd.read_csv(location2_file1)
    data_test = pd.read_csv(location2_file2)
    
# %% Data Wrangling: we want one dataset with monthly data (ie average the data within each month), normalize the data, then resplit to train and test
combined_data = pd.concat([data_train, data_test], ignore_index=True) #ignores index reindexes everything, I checked and there are no null values
#columns are: date (yyyy-mm-dd, object, I need to change its type), meantemp(float, I think in C), humidity(float, a %), wind_speed (float), meanpressure (float), use combined_data.dtypes to see
combined_data['date'] = pd.to_datetime(combined_data['date'], format='%Y-%m-%d') #converts the date to a datetime datatype
monthly_data = combined_data.groupby(combined_data['date'].dt.to_period('M')).mean() #this dt.to_period('M') keeps the date term up to month, ie year and month are saved, and days are dropped, then it groups by month and takes the mean of the month (I couldn't find a way to see the groupby step?)
# Normalize each column to the range [0, 1]
normalized_data = monthly_data.copy()
# normalization options:
# min/max scaling ends between [0 1]: apply(lambda x: (x - x.min()) / (x.max() - x.min()), to reverse I need to save min and max
# Z-score normalization, data has a mean of 0 and std of 1, this is what we have used in the past:  normalized_data.iloc[:, 1:] = (normalized_data.iloc[:, 1:] - mean) / std , need to save the mean and std to reverse
# I am going to try min/max scaling so that the values are all still positive as we don't have any negative temps
mean_vals = normalized_data.iloc[:,1:].mean()
std_vals = normalized_data.iloc[:,1:].std()
normalized_data.iloc[:, 1:] = (normalized_data.iloc[:, 1:] - mean_vals) / std_vals  #excludes the first (date) column, then, apply is a method that applies a function along an axis of the dataframe, the lambda function takes the current column, subtracts the min value of the column and divides by the difference between max and min
unique_months = normalized_data.index.to_timestamp().to_period('M').unique()
month_to_number = {month: i + 1 for i, month in enumerate(unique_months)}
normalized_data['date_numeric'] = normalized_data.index.to_timestamp().to_period('M').map(month_to_number) #for an explanation of these lines (made by and explained by AI, both the lines and explanation), see data_wrangling_explanation.txt
split_date = '2014-09' 
monthly_train = normalized_data[normalized_data['date'] < split_date ] #data through 2014-08
monthly_test = normalized_data[normalized_data['date'] >= split_date ]

# %% Create the Neural net that will hopefully represent the ODEs
# ODE function class so it can be used with each weather variable (I guess)
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
        
    def ode_func(self, t, y):
        return self.network(y) #return dy/dt

    def forward(self, y0, tsteps):
        yhat = odeint(self.ode_func, y0, tsteps) #! pull in tsteps as it will have to change sizes
        return yhat

def train(ytrain, ttrain, model, optimizer, lossfn):
    
    model.train()
    optimizer.zero_grad()
    yhat = model(ytrain[0,:],ttrain)
    loss = lossfn(yhat,ytrain)
    loss.backward()
    optimizer.step()
    
    return loss.item()

# %% setup data 
t_train = torch.tensor(np.array(monthly_train['date_numeric'].iloc[:]), dtype=torch.float64) #.reshape(20,1) #[20,1]
t_test = torch.tensor(np.array(monthly_test['date_numeric'].iloc[:]), dtype=torch.float64) #[32,1]
y_train = torch.tensor(np.array(monthly_train.iloc[:,1:-1]), dtype=torch.float64) # [20,4]
y_test = torch.tensor(np.array(monthly_test.iloc[:,1:-1]), dtype=torch.float64) # [32,4]
t_everything = torch.cat((t_train, t_test))

#initialize the model 
hlayers = 3
hwidth = 40
num_inputs = y_train.shape[1]
model = ODEFunc(num_inputs, hlayers, hwidth)
model.double()
# print(model.network[2].weight.shape)
# optimizer = torch.optim.Adam(model.parameters(), lr = .1)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

# %% Setup function to train on only part of the data at a time
# I can do two loops outer will change how many data points the inner epoch (normal training loop) has. Then add more each outer loop etc. I can plot loss over everything, I need to figure out time and odeint stuff etc.
train_indexes = np.array([4,8,12,16,20])
epochs = 400 #np.array([50,200,150,150])
lossfn = nn.MSELoss()
losses = []
for i in range(4):
    
    if i > 0:
        model.eval()
        ycheck = model(y_train[1,:], t_train)
        plt.figure()
        plt.plot(losses)
        plt.figure()
        plt.plot(t_train, y_train.detach().numpy()[:, 1], 'b', label='True', linestyle='None', marker = 'o')
        plt.plot(t_train, ycheck.detach().numpy()[:, 1], 'b--', label='Training Fit')
        plt.show()
        plt.pause(0.001)
        model.train()
        
    print('Training on ', i)
    
    for e in range(epochs): #range(epochs[i]):
        losses.append(train(y_train[0:train_indexes[i]], t_train[0:train_indexes[i]], model, optimizer, lossfn))
#%%
y_all_units = monthly_data.iloc[:,1:]
# %% Plotting
#loss plot
plt.figure()
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
 
# Verification calculations and plot
t_all = normalized_data['date']
y_all = monthly_data.iloc[:,1:5]
model.eval()
with torch.no_grad():
    yhat = model(y_train[0, :], t_train)
    ytake2 = model(y_train[0,:], t_everything) #technically this should run on y_test[0,:], t_test
    #unnormalize with mean_vals and std_vals from z normalization
    yhat = yhat.detach().numpy()*std_vals.values + mean_vals.values
    ytake2 = ytake2.detach().numpy()*std_vals.values + mean_vals.values
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))
    axs[0].plot(t_all, y_all.iloc[:, 0], 'r', label='True', linestyle='None', marker = 'o')
    axs[0].plot(t_all[:20], yhat[:, 0], 'r-', label='Training Fit')
    axs[0].plot(t_all, ytake2[:,0], 'r.-.', label = 'Test Output')
    
    axs[0].set_title('Mean Temp')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Temperature (C)')
    axs[0].legend()
    
    axs[1].plot(t_all, y_all.iloc[:, 1], 'b', label='True', linestyle='None', marker = 'o')
    axs[1].plot(t_all[:20], yhat[:, 1], 'b-', label='Training Fit')
    axs[1].plot(t_all, ytake2[:,1], 'b.-.', label = 'Test Output')
    axs[1].set_title('Humidity')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Humidity (g/m^3)')
    axs[1].legend()
    
    axs[2].plot(t_all, y_all.iloc[:, 2], 'g', label='True', linestyle='None', marker = 'o')
    axs[2].plot(t_all[:20], yhat[:, 2], 'g-', label='Training Fit')
    axs[2].plot(t_all, ytake2[:,2], 'g.-.', label = 'Test Output')
    axs[2].set_title('Wind Speed')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Wind Speed (m/s)')
    axs[2].legend()
    
    axs[3].plot(t_all, y_all.iloc[:, 3], 'orange', label='True', linestyle='None', marker = 'o')
    axs[3].plot(t_all[:20], yhat[:, 3], 'orange', linestyle = '-', label='Training Fit')
    axs[3].plot(t_all, ytake2[:,3], 'orange', linestyle = '-.', label = 'Test Output')
    axs[3].set_title('Mean Pressure')
    axs[3].set_xlabel('Date')
    axs[3].set_ylabel('Mean Pressure (hPa)')
    axs[3].set_ylim(990,1030)
    axs[3].legend()
    
    plt.tight_layout()
    plt.show()


# %%
