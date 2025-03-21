#%% Import packages 
import numpy as np
from pysr import PySRRegressor
# %% load data 
data = np.loadtxt('subset.csv', delimiter=',')
outputs = data[:, 0]
inputs = data[:1000, 1:]
# %% setup symbolic regression
model = PySRRegressor(
    niterations=25,
    binary_operators=["+", "*"],
    unary_operators=["cos", "exp", "sin"],
    procs=4,
)
# %% fit the model
model.fit(inputs,outputs)
print(model)
# %%
