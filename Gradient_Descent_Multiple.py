import numpy as np
#Eqn
#delta-W = eta*Error_term*x(i).
#where
#   delta-W - change in weight to be applied.
#   x(i) -- Input
#   eta = Learnrate
#   Error_term = error*derivative of activation fn (f(h)), called f'(h).
#   error - (y-y^)
#   Y - label
#   y^i - activation_function(h)
#   f(h) - sigmoid for our example
#   where h = w1x1+w2x2
#   f'(h) - derivative of f(h) , also called output gradient.
#   Intuitively, error*gradient will show whether error is reducing or not and the idea is to minimize it.

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    # Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

### Calculate one gradient descent step for each weight
### Note: Some steps have been consilated, so there are
###       fewer variable names than in the above sample code

# TODO: Calculate the node's linear combination of inputs and weights
h = np.dot(x,w)

# TODO: Calculate output of neural network
nn_output = sigmoid(h)

# TODO: Calculate error of neural network
error = y-nn_output

# TODO: Calculate the error term
#       Remember, this requires the output gradient, which we haven't
#       specifically added a variable for.
error_term = error*sigmoid_prime(h)

# TODO: Calculate change in weights
del_w = learnrate*error_term*x

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)