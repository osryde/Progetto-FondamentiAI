# Questo modulo si occupa di gestire i dataset (MNIST)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Trasfoma le immagini in tensori in quando definiti su una scala di grigi 28x28
# ToTensor() - converte le immagini in tensori (1,28,28) con valori tra 0 e 1
# Normalize() - normalizza i tensori con media e deviazione standard
# batch_size - numero di immagini per batch con label, cioè il numero di immagini che vengono processate insieme
def get_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),                         # converte immagine PIL in tensor (range 0-1)
        transforms.Normalize((0.1307,), (0.3081,))     # normalizza: (media, std) per MNIST
    ])

    # Carica il dataset MNIST per il training e il test nella cartella 'data'
    train = datasets.MNIST(
        root='../data', train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        root='../data', train=False, download=True, transform=transform
    )

    # DataLoader - permette di iterare sui dataset in batch, suddividendo le immagini in batch di dimensione batch_size
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
