# File contenente il modello della rete neurale MLP (Multi-Layer Perceptron)
import torch
# Moduli per la creazione di layer
import torch.nn as nn 
# Funzioni di attivazione
import torch.nn.functional as F

# MLP - Multi-Layer Perceptron parametrizzato
# nn.Module - classe base per tutti i modelli di PyTorch
class MLPFlexible(nn.Module):

    # Costruttore della rete neurale
    def __init__(self, input_size=784, hidden_sizes=[128], num_classes=10, dropout_prob=0.0):
        super().__init__()
        
        # TODO: Applicare droput anche sull'input layer?
        
        layers = []
        in_size = input_size
        
        # Costruiamo i layer nascosti
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size)) # Layer lineare con input in_size e output hidden_size
            layers.append(nn.ReLU()) # Aggiungiamo la funzione di attivazione ReLU
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            in_size = hidden_size # Aggiorniamo la dimensione di input per il prossimo layer
        
        # Layer di output
        layers.append(nn.Linear(in_size, num_classes))
        
        # Mettiamo tutto in sequenza
        self.model = nn.Sequential(*layers) # unpacking per passare la lista di layer come argomenti
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.model(x)
        return x
