#=
HW2.jl
Jacob Child
Jan 23rd, 2025
Big Goal: I want to understand the neural net I made in python, and reimplement it in julia to do so. 
=#
using DelimitedFiles

# Load the data
data = readdlm("DataSets/HW1Data/auto-mpg.data", header=false) # 398 x 9 matrix 
data = data[:,1:8] # 398 x 8 matrix, dropped names 
# look for any '?' and delete that row
bad_rows = findall(x->x=="?", data)
bad_row_indices = [index[1] for index in bad_rows]
data = data[setdiff(1:end, bad_row_indices), :] # 392 x 8 matrix
# convert all to floats
data = convert(Array{Float64,2}, data)
#shuffle the data
nrows = size(data, 1)
shuffled_indices = rand(1:nrows, nrows)
data = data[shuffled_indices, :]
# split the data into training and testing
train_data = data[1:round(Int, 0.8*nrows), :]
test_data = data[round(Int, 0.8*nrows)+1:end, :]
Xtrain = train_data[:, 2:end]
Ytrain = train_data[:, 1]
Xtest = test_data[:, 2:end]
Ytest = test_data[:, 1]

# Normalize the data
