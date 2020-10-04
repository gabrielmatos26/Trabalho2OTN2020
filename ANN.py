import numpy as np

class ANN:
    def __init__(self, weights, num_hidden_layers=1, hidden_size=128):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.weights = weights
    
    def predict(self, x):
        out = np.concatenate((np.array([1]), x.copy()))
        for i in range(self.num_hidden_layers):
            out = np.tanh(out @ self.weights[i])
            
            out = np.concatenate((np.array([1]), out))
            
        out = out @ self.weights[-1]
        return np.tanh(out)