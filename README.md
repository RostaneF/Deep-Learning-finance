# Deep-Learning-finance
Using deep learning our purpose will be to implement a stock/bond picking portfolio optimization.

## PART 1 : Implementing a neural_network algorithm and testing it

### Neural Network Implementation using Dense Layers and Activation Functions

This code provides an implementation of a neural network using dense layers and activation functions. It allows you to create a multi-layer neural network and train it on a given dataset. The code is written in Python and utilizes the NumPy and Matplotlib libraries.

The main components of the code are as follows:

1. **Layer class**: This is the base class that defines the basic characteristics of a layer. All other layers inherit from this class. It contains methods for forward propagation and backward propagation.

2. **Dense class**: This class represents a dense layer in the neural network. It inherits from the Layer class and defines the forward and backward propagation methods specific to dense layers. It includes attributes for weights and biases.

3. **Activation class**: This class represents an activation layer in the neural network. It also inherits from the Layer class and defines the forward and backward propagation methods specific to activation layers. It includes attributes for the activation function and its derivative.

4. **Tanh class**: This class represents the hyperbolic tangent activation function. It inherits from the Activation class and provides the implementation of the tanh activation function and its derivative.

5. **Other activation functions**: You can define additional activation functions by creating new classes similar to the Tanh class.

6. **Mean Squared Error (MSE) functions**: The code includes an implementation of the mean squared error (MSE) function and its derivative, which are used as metrics for estimating the deviation between the network's output and the expected output.

7. **Neural Network class**: This class represents the overall neural network. It allows you to add layers to the network, train the network on a given dataset, and make predictions using the trained network.

8. **Example: XOR function**: The code includes an example that demonstrates how to use the neural network to learn the XOR function. It constructs a neural network with two dense layers and two tanh activation layers. It trains the network on the XOR dataset and then makes predictions using the trained network.

By running the `main()` function, you can see the training progress and the network's predictions on the XOR dataset. The code also includes a function `format_predict()` to format the predictions and a function `XOR()` to verify the results.

Please note that the code assumes the availability of the NumPy and Matplotlib libraries.
