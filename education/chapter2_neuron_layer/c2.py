#---------------------------------------------------------------------------------
#----WE ARE LOOKING AT A NEURON, OUTPUT LAYER SOMEWHERE IN THE NEURAL NETWORK-----
#---------------------------------------------------------------------------------
# neurons in the previous layer acting as input
inputs = [1, 2, 3, 2.5]

# each of those inputs input has an associated weight
# (how much we care about that input)
# but this time, those inputs have different sets of associated weight,
# depending on the output neuron it is being fed into
weight_set1 = [0.2, 0.8, -0.5, 1.0]
weight_set2 = [0.5, -0.91, 0.26, -0.5]
weight_set3 = [-0.26, -0.27, 0.17, 0.87]

# every unique neuron has a unique bias
# (how much this neuron matters)
# but this time we have three output neurons
bias1 = 2
bias2 = 3
bias3 = 0.5

# the total output value that this neuron will pass on to the next layer
# since we are dealing with an output layer, we expect three items;
# each item taking a sum of four input-weight dot    product, one from each input
output = [
    inputs[0] * weight_set1[0] + inputs[1] * weight_set1[1] + inputs[2] * weight_set1[2] + inputs[3] * weight_set1[3] + bias1,
    inputs[0] * weight_set2[0] + inputs[1] * weight_set2[1] + inputs[2] * weight_set2[2] + inputs[3] * weight_set2[3] + bias2,
    inputs[0] * weight_set3[0] + inputs[1] * weight_set3[1] + inputs[2] * weight_set3[2] + inputs[3] * weight_set3[3] + bias3,
]
print(output)
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

# how do we adjust weights and biases for future inputs?-->