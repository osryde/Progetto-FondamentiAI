import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Modello salvato in {path}")

def load_model(model, path, device):
    # State_dict - dizionario che contiene i pesi del modello
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval() # ; imposta il modello in modalità valutazione (disabilita dropout e batch norm)
    print(f"Modello caricato da {path}")
