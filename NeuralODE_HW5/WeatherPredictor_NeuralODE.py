#NeuralODE_HW5.py
#Jacob Child
#Feb 10, 2025

#! .venv\Scripts\Activate.ps1

# %% Import packages etc
import pandas as pd
import numpy as np
from torchdiffeq import odeint_adjoint as odeint # this is the recommended ode solver for backprop reasons, it takes a function that is an nn.module, ie odeint(func, y0, t) where func is the neural net or nn.module
import torch 
from torch import nn
import torch.optim as optim
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
min_vals = normalized_data.iloc[:,1:].min()
max_vals = normalized_data.iloc[:,1:].max()
normalized_data.iloc[:, 1:] = normalized_data.iloc[:, 1:].apply(lambda x: (x - min_vals[x.name]) / (max_vals[x.name] - min_vals[x.name])) #excludes the first (date) column, then, apply is a method that applies a function along an axis of the dataframe, the lambda function takes the current column, subtracts the min value of the column and divides by the difference between max and min
split_date = '2014-09' 
monthly_train = monthly_data[monthly_data['date'] < split_date ] #data through 2014-08
monthly_test = monthly_data[monthly_data['date'] >= split_date ]

# %% Create the Neural net that will hopefully represent the ODEs
# ODE function class so it can be used with each weather variable (I guess)
class ODEFunc(nn.Module):
    def __init__(self, hidden_layers, hidden_width):
        super(ODEFunc, self).__init__()
        
        layers = []
        layers.append(nn.Linear(2,hidden_width))
        layers.append(nn.Tanh()) #? I may need to change the activation function
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Tanh()) 
            
        layers.append(nn.Linear(hidden_width,1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)
