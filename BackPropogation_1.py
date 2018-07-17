import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

#shape 1,3
x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

#shape = 3,2
weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

#shape = 1,2
weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate output error
error = target - output

# TODO: Calculate error term for output layer
output_error_term = error*(sigmoid(output_layer_in)*(1-sigmoid(output_layer_in)))

# TODO: Calculate error term for hidden layer
hidden_error_term = output_error_term*(sigmoid(hidden_layer_input)*(1-sigmoid(hidden_layer_input)))*weights_hidden_output

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate*output_error_term*hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_wt1=learnrate*hidden_error_term[0]*x
delta_wt2=learnrate*hidden_error_term[1]*x
delta_w_i_h = np.array([delta_wt1,delta_wt2])
delta_w_i_h2 = learnrate*hidden_error_term*x[:,None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
