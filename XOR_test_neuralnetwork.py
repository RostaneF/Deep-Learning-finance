import numpy as np
import os
new_path = r"V:\Rostane\Deep Learning"
os.chdir(new_path)
from first_perceptron import NeuralNetwork, Dense, Tanh
import matplotlib.pyplot as plt

def format_predict(pred): # Add this function to adapt the format of predictions
    new_pred = []
    for i in pred:
        elt = 0 if i[0][0] < 0.5 else 1 
        new_pred.append(elt)
    return new_pred

def XOR(data): # Implement a simple function to verify the results
    results = []
    for couple in data:
        results.append(couple[0][0] ^ couple[1][0]) # The ^ operator corresponds to XOR in Python
    return results

def main():
    # To verify that the constructed perceptron learns well, we make it learn XOR, the exclusive OR
    errors = []
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
    
    # Construct the neural network 
    network = NeuralNetwork()
    network.add(Dense(2, 3))
    network.add(Tanh())
    network.add(Dense(3, 1))
    network.add(Tanh())
    
    # Training: 
    epochs = 150
    learning_rate = 0.1
    network.train(X, Y, epochs, learning_rate)
    
    # Prediction: 
    input_data = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    pred = network.predict(input_data)
    verification = XOR(input_data.tolist())
    print(f'\nData: {input_data.tolist()} \nPredictions on your data: {format_predict(pred)}\nVerification by XOR function: {verification}')
        
main()
