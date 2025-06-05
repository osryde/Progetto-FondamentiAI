import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # barra di progresso

def train(model, train_loader, optimizer, criterion, device):
    model.train()  # abilita modalità training (dropout attivo)
    
    # Loss media per i batch
    running_loss = 0.0
    
    # Il training funziona così:
    # 1. Passiamo gli input nel modello (forward) per ottenere le predizioni.
    # 2. Calcoliamo la loss confrontando predizioni e target veri.
    # 3. Calcoliamo i gradienti della loss rispetto ai pesi (backward).
    # 4. Aggiorniamo i pesi usando i gradienti (optimizer.step).
    # 5. Ripetiamo su tutti i batch per migliorare il modello.
    for inputs, targets in tqdm(train_loader):

        # Sposta i dati sul dispositivo (GPU o CPU)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # resetta gradienti
        outputs = model(inputs) # forward pass chiamata in automatico del metodo forward del modello
        
        loss = criterion(outputs, targets)  # calcola la loss
        loss.backward()                # backward pass: calcola gradienti
        optimizer.step()               # aggiorna i pesi
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)
