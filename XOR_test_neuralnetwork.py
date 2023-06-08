import numpy as np
import os
nouveau_chemin = r"V:\Rostane\Deep Learning"
os.chdir(nouveau_chemin)
from first_perceptron import neural_network, Dense, Tanh
import matplotlib.pyplot as plt

def format_predict(pred): # On ajoute cette fonction pour adapter le format des predictions
    new_pred = []
    for i in pred :
        elt = 0 if i[0][0] < 0.5 else 1 
        new_pred.append(elt)
    return new_pred

def XOR(data): # On implémente une fonction simple pour vérifier les resultats
    results = []
    for couples in data:
            results.append(couples[0][0]^couples[1][0]) #l'opérateur ^ correspond au XOR sous python
    return results

def main():
    #Pour vérifier que le perceptron que nous avons construit apprend bien, On le fait apprendre sur XOR : le OU exclusif
    errors = []
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
    
    # Construction du reseau de neuronnes 
    network = neural_network()
    network.add(Dense(2,3))
    network.add(Tanh())
    network.add(Dense(3,1))
    network.add(Tanh())
    
    #Apprentissage : 
    epochs = 150
    learning_rate = 0.1
    network.train(X, Y, epochs, learning_rate)
    
    #Prédiction : 
    Input_data = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    pred = network.predict(Input_data)
    verif = XOR(Input_data.tolist())
    print(f'\nDonnées : {Input_data.tolist()} \nPrédiction sur vos données : {format_predict(pred)}\nVerification par la fonction XOR : {verif}')
        
main()