import numpy as np
import matplotlib.pyplot as plt

"""Creating the Layer class corresponds to step 2 of the proposed plan, where we create a "Layer" class that defines the basic characteristics that all other layers will inherit."""
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        # Define the forward propagation here to obtain the output of the layer as a function of the input data
        pass
    
    def backward(self, output_gradient, learning_rate): # For simplicity, a learning rate is considered as a parameter for now.
        # In the next version, optimizers will be considered for this metric. 
        # Update the parameters using gradient descent
        pass
  
"""Once we have created the Layer class, we create the Dense layer class that inherits the fundamental properties from the parent class and is defined as follows:"""

class Dense(Layer):
    
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) # Weight matrix of dimensions j x i (see)
        self.bias = np.random.randn(output_size, 1) # Column vector of biases
        
    def forward(self, input): # Define the forward propagation for the dense layer
        self.input = input # Associate the inputs that we want to pass to the layer's class data
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output   # Implement the propagation as defined: Y = WX + B. Note that np.dot is used for matrix dot product
    
    def backward(self, output_gradient, learning_rate):
        # In this method, we calculate the derivatives of the error with respect to the weights, biases, and input. 
        weights_gradient = np.dot(output_gradient, self.input.T) # dE/dW = dE/dY.dY/dW = dE/dY.X^T 
        # Update the parameters using gradient descent.
        self.weights -= weights_gradient * learning_rate # w* = w - learning_rate * dE/dW
        self.bias -= learning_rate * output_gradient # From the calculation, we obtained that dE/dB = dE/dY.dY/dB = dE/dY.
        return np.dot(self.weights.T, output_gradient)  # The backward propagation returns the derivative of the error with respect to the input. 
                                                        # By calculation, we determined that dE/dX = dE/dY.dY/dX = W^T*dE/dY
    
"""Now that we have implemented the Layer class and the Dense class for dense layers, let's implement a class for the activation layer.""" 

class Activation(Layer):
    def __init__(self, activation, activation_prime): 
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):  # The forward propagation here simply applies the activation function defined as a class attribute to each incoming neuron.
        self.input = input
        self.output = self.activation(input)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # Use useful properties of numpy tools
        return np.multiply(output_gradient, self.activation_prime(self.input)) # dE/dX = dE/dY(x)f'(X)
    

"""Finally, we proceed with the implementation of activation functions and error metrics (to estimate the deviation of the output from the expected targets):""" 

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)
        
### Define other activation functions here ###

###                                             ###

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


### Now we can create a Neural Network model to generate a network of neurons by adding layers, activation functions, etc.: ###
    
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def get_layers(self):
        return self.layers
    
    def train(self, X_train, Y_train, epochs, learning_rate):
        errors = []
        # Training the neural network:
        for e in range(epochs):
            error = 0
            for x, y in zip(X_train, Y_train): 
                output = x 
                for layer in self.layers:
                    # Pass each input through each layer of the neural network
                    output = layer.forward(output)
                
                # Once we have obtained the output, calculate the MSE, which will be used as an input for the backward propagation.
                error += mse(y, output)
                
                grad = mse_prime(y, output)
                for layer in reversed(self.layers): # Similar to forward propagation, pass the output through each layer of the network in reverse order to refine the parameters
                    grad = layer.backward(grad, learning_rate)
            
            error /= len(X_train)
            errors.append(error)
            print('%d/%d, error = %f' % (e + 1, epochs, error), f"output = {output}")
        plt.plot(errors)
        plt.xlabel("Epochs")
        plt.ylabel("Error (MSE)")
        plt.title("Variation of Error during Training")
        plt.show()
        return
    
    def predict(self, input_data):
        predictions = []
        for x in input_data:
            output = x
            for layer in self.layers:
                output = layer.forward(output)
            predictions.append(output)
        return predictions
        
