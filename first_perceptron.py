import numpy as np
import matplotlib.pyplot as plt

""" La création de la classe layer correspond à l'étape 2 du plan proposé, on crée une classe "Layer" qui definit les caractéristiques de bases dont toutes les autres couches hériteront"""
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self,input):
        # On définit ici la forward propagation pour obtenir l'output de la couche généré comme fonction image de la donnée d'input
        pass
    
    def backward(self, output_gradient, learning_rate): # Par soucis de simplicité, on considère un Learning rate passé en paramètre pour le moment.
        # Dans la version suivante, on considerera des optimiseurs pour cette metrique. 
        # Mettre à jour les paramètres en passant par la descente de gradient
        pass
  
""" Une fois que l'on a créé la classe Layer : On crée la classe des couches denses qui hérite des propriétés fondamentales de la classe mère et que l'on définit comme suit : """

class Dense(Layer):
    
    def __init__(self,input_size,output_size):
        self.weights = np.random.randn(output_size, input_size) # Matrice de poids de dimensions j x i (voir)
        self.bias = np.random.randn(output_size, 1) # Vecteur colonne de biais
        
    def forward(self, input): # On définit la forward propagation dans le cadre de la couche dense
        self.input = input # On associe les inputs que l'on souhaite passer à la donnée de classe de la couche
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output   # On implémente la propagation comme définie : Y = WX + B. à noter qu'on utilise np.dot pour le produit scalaire matriciel
    
    def backward(self, output_gradient, learning_rate):
        # Dans cette methode, on calcule les dérivées de l'erreur par rapport aux poids, au biais ainsi que de l'input. 
        weights_gradient = np.dot(output_gradient, self.input.T) # dE/dW = dE/dY.dY/dW = dE/dY.X^t 
        # Par la descente de gradient, on met à jour les paramètres.
        self.weights -= weights_gradient*learning_rate # w* = w - learning_rate * dE/dW
        self.bias -= learning_rate*output_gradient # Par le calcul on a obtenu que dE/dB = dE/dY.dY/dB = dE/dY.
        return np.dot(self.weights.T,output_gradient)  # La backward propagation renvoie le dérviée de l'erreur par rapport à l'input. 
                                                        # Par le calcul, on a determiné que dE/dX = dE/dY.dY/dX = W^t*dE/dY
    
""" Maintenant que nous avons implémenté la classe layer et la classe Dense pour les couches denses, on implémente une classe pour la couche d'activation.""" 

class Activation(Layer):
    def __init__(self, activation, activation_prime): 
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):  # La forward propagation consiste ici simplement à appliquer à chaque neuronne entrant, la fonction d'activation définie en attribut de classe
        self.input = input
        self.output = self.activation(input)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # On utilise les proprités utiles des outils de numpy 
        return np.multiply(output_gradient, self.activation_prime(self.input)) # dE/dX = dE/dY(x)f'(X)
    

""" Pour finir, on procède à l'implémentation des fonctions d'activation et metriques d'erreurs (pour estimer l'écart de la sortie par rapport aux objectifs attendus:""" 

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1-np.tanh(x)**2
        super().__init__(tanh,tanh_prime)
        
### Définir d'autres fonctions d'activation ici ### 

###                                             ###
        
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true,y_pred):
    return 2*(y_pred - y_true)/np.size(y_true)


### On peut desormais créer un modèle de classe Neural network permettant de générer un réseau de neuronnes en ajoutant des couches, fonctions d'activation.. etc :
    
class neural_network:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def get_layers(self):
        return self.layers
    
    def train(self, X_train, Y_train, epochs, learning_rate):
        errors = []
        # Entrainement du réseau de neuronnes :
        for e in range(epochs):
            error = 0
            for x, y in zip(X_train, Y_train): 
                output = x 
                for layer in self.layers:
                    # On fait passer chaque input par chacune des couches du réseau de neuronnes
                    output = layer.forward(output)
                
                #Une fois que nous avons obtenu l'output, on calcule la MSE qui servira d'input à la backward propagation.
                error+= mse(y, output)
                
                grad = mse_prime(y, output)
                for layer in reversed(self.layers): # De même que pour la forward Propagation, on fait passer l'output par chacune des couches du reseau dans le sens inverse pour affiner les paramètres
                    grad = layer.backward(grad, learning_rate)
            
            error /= len(X_train)
            errors.append(error)
            print('%d/%d, error = %f' % (e + 1, epochs, error), f"output = {output}")
        plt.plot(errors)
        plt.xlabel("Epochs")
        plt.ylabel("Erreur (MSE)")
        plt.title("Variation de l'erreur au cours l'apprentissage")
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
        