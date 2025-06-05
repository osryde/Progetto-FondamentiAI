import torch
import torch.nn.functional as F

def test(model, test_loader, device):
    model.eval()  # modalità valutazione (dropout disattivato)
    correct = 0
    total = 0
    loss_total = 0

    with torch.no_grad():  # disabilita il calcolo dei gradienti
        for inputs, targets in test_loader:
            # Sposta i dati sul dispositivo (GPU o CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs) # Chiamata in automatico del metodo forward del modello
            # La cross-entropy loss confronta le probabilità predette con i target veri
            loss = F.cross_entropy(outputs, targets)

            # Trova la classe con probabilità massima tra le predizioni
            _, predicted = torch.max(outputs, dim=1)

            # Conta quante predizioni corrispondono ai target veri
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            loss_total += loss.item() # Accumula la loss del batch

    accuracy = correct / total
    avg_loss = loss_total / len(test_loader)
    return accuracy, avg_loss
