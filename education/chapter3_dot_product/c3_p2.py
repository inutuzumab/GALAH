#---------------------------------------------------------------------------------
#----WE ARE LOOKING AT A NEURON, OUTPUT LAYER SOMEWHERE IN THE NEURAL NETWORK-----
#----BUT WE ARE MAKING IT NICER USING NUMPY---------------------------------------
#----EXAMPLE 1: n-D VECTORS--------------------------------------------------------
#---------------------------------------------------------------------------------
import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, -0.17, -0.87]
]
bias = [2, 3, 0.5]

# weights must be before inputs
# otherwise: shapes (4,) and (3,4) not aligned: 4 (dim 0) != 3 (dim 0)
# weights precedes inputs in this case ONLY because we need the 1x4 matrix to be first.
output = np.dot(weights, inputs) + bias;
print(output)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------