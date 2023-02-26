#---------------------------------------------------------------------------------
#--------WE ARE LOOKING AT A SINGLE NEURON SOMEWHERE IN THE NEURAL NETWORK--------
#---------------------------------------------------------------------------------
# neurons in the previous layer acting as input
inputs = [1, 2, 3]

# each of those inputs input has an associated weight
# (how much we care about that input)
weights = [0.2, 0.8, -0.5]

# every unique neuron has a unique bias
# (how much this neuron matters)
bias = 2

# the total output value that this neuron will pass on to the next layer
output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
print(output)
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------