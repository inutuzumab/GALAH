#---------------------------------------------------------------------------------
#----BATCHES, LAYERS, AND OBJECTS: MODELLING n LAYERS OF NEURONS------------------
#---------------------------------------------------------------------------------
import numpy as np

# Previously, we've had an array if samples (of inputs)...
# We want to now pass in a GROUP of inputs at a time;
# Batch size is the number of samples you pass in each time.
# If we are training our neural network based on an epoch of e samples
# (remember; training = tuning),
# Then our batch size is the partition of the epoch we are passing through each time.
# TOO SMALL: there is a lot of noise... tuning to EVERY single point strengthens outliers.
# TOO LARGE: computers are generally not strong enough to handle this... also too generalised.
inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0 ],
    [-1.5, 2.7, 3.3, -0.8]
]
'''
Weights transposed to be multiplied with inputs:
Output1     Output2     Output3     <-- THE WEIGHTINGS ASSOCIATED WITH EACH OUTPUT
[[ 0.2        0.5        -0.26]
 [ 0.8       -0.91       -0.27]    
 [-0.5        0.26        0.17] 
 [ 1.0       -0.5         0.87]]
 
 Multiplied as:
[I_row_1 • W_col_1     I_row_1 • W_col_2       I_row_1 • W_col_3
 I_row_2 • W_col_1     I_row_2 • W_col_1       I_row_2 • W_col_1
 I_row_3 • W_col_1     I_row_3 • W_col_1       I_row_3 • W_col_1]

The unique bias of each unique output is then added for that output.
The bias is a row vector, so just add to each row.

!! 
Each row represents the values of the outputs for ONE (1) SAMPLE
Each column represents the samples that together constitute the BATCH
 '''
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5], 
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2, 3, 0.5]

weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33], 
    [-0.44, 0.73, -0.13]
]
biases2 = [-1, 2, -0.5]
# note that now inputs comes before weights... see matrix multiplication image
# the bias of each output is added to each output row
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer1_outputs)
print(layer2_outputs)
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
'''Note, consider:
inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.],
    [-1.5, 2.7, 3.3, -0.8]
]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, -0.17, -0.87]
]
bias = [2, 3, 0.5]
output = np.dot(weights, inputs) + bias;
print(output)

This will not work because np.dot treats them like a matrix. So: 1.0 * 0.2, 2.0 * 0.5, 3.0 * -0.26, and 2.5 * undefined
'''