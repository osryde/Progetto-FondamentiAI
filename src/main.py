import torch

# TODO: Sistemare CUDA
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.current_device = lambda: 0
torch.cuda._initialized = True

from torch import nn
from data_loader import get_loaders
from model import MLPFlexible
from train import train
from test import test
from save_model import save_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Iperparametri
    batch_size = 64
    learning_rate = 0.001
    epochs = 10
    hidden_sizes = [512,512,256,128]       # Strati nascosti
    dropout_prob = 0.4           # Dropout 

    # Carica i dati
    train_loader, test_loader = get_loaders(batch_size=batch_size)

    # Crea il modello parametrico con dropout
    model = MLPFlexible(
        input_size=28*28,
        hidden_sizes=hidden_sizes,
        num_classes=10,
        dropout_prob=dropout_prob
    ).to(device)

    # Ottimizzatore
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    save_model(model, "mlp_flexible_dropout.pth")

    # Test
    accuracy, avg_loss = test(model, test_loader, device)
    print(f"Test Accuracy: {accuracy*100:.2f}%, Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
