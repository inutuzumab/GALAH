#---------------------------------------------------------------------------------
#----WE ARE LOOKING AT A NEURON, OUTPUT LAYER SOMEWHERE IN THE NEURAL NETWORK-----
#----BUT WE ARE MAKING IT NICER USING NUMPY---------------------------------------
#----EXAMPLE 1: 1D VECTORS--------------------------------------------------------
#---------------------------------------------------------------------------------
import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = np.dot(weights, inputs) + bias;
print(output)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
'''
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)
'''