import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from model import MLPFlexible  # Assumendo che tu abbia una definizione flessibile del modello
from train import train  # Funzione di training definita in train.py
from test import test  # Funzione di test definita in test.py
from data_loader import get_loaders  # Funzione per ottenere i data loader, definita in data_loader.py

def plot_accuracy_loss():
    """
    Funzione principale per il training, test e visualizzazione delle performance dei modelli MLP
    con e senza dropout per entrambe le configurazioni (MLP a 1 strato nascosto e multi strati nascosti).
    
    Vengono creati i seguenti plot:
    1. Accuratezza di test in funzione del valore di dropout.
    2. Confronto tra il loss di training e di validazione (test) in funzione del dropout.
    3. Analisi dell'overfitting, confrontando l'accuratezza di training e test.

    I risultati vengono salvati in un file CSV e i plot in una cartella separata chiamata 'plots'.
    """
    
    # Impostazione del dispositivo (GPU o CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parametri di training
    batch_size = 64
    learning_rate = 0.001
    epochs = 10
    hidden_sizes_single = [128]  # Configurazione per il MLP a 1 strato nascosto
    hidden_sizes_multi = [512, 512, 256, 128]  # Configurazione per il MLP a più strati nascosti

    # Valori di dropout da sperimentare
    dropout_values = [0.0, 0.2, 0.4, 0.6, 0.8]

    # Ottenimento dei data loader per il training e il test
    train_loader, test_loader = get_loaders(batch_size=batch_size)

    # Lista per memorizzare i risultati da plotare
    results = []

    # Ciclo attraverso i vari valori di dropout per entrambe le configurazioni (1 layer e multi layer)
    for dropout in dropout_values:
        for model_type, hidden_sizes in [('Single Layer', hidden_sizes_single), ('Multi Layer', hidden_sizes_multi)]:
            
            # Creazione del modello con il dropout specificato
            model = MLPFlexible(input_size=28*28, hidden_sizes=hidden_sizes, output_size=10, dropout_rate=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Training e test del modello
            train_acc, train_loss = train(model, train_loader, optimizer, epochs, device)
            test_acc, test_loss = test(model, test_loader, device)

            # Memorizzazione dei risultati per ciascun valore di dropout
            results.append({
                'dropout': dropout,
                'model_type': model_type,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'test_loss': test_loss
            })

    # Salvataggio dei risultati in un file CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results_performance.csv', index=False)

    # Creazione della cartella per i plot se non esiste
    os.makedirs('plots', exist_ok=True)

    # ---- PLOT 1: Accuratezza di test vs Dropout ----
    plt.figure(figsize=(8, 6))
    for model_type in ['Single Layer', 'Multi Layer']:
        model_results = results_df[results_df['model_type'] == model_type]
        plt.plot(model_results['dropout'], model_results['test_acc'], marker='o', label=f'{model_type} Test Accuracy')

    plt.title('Test Accuracy vs Dropout')  # Titolo del grafico
    plt.xlabel('Dropout Rate')  # Etichetta asse x (valori di dropout)
    plt.ylabel('Test Accuracy')  # Etichetta asse y (accuratezza di test)
    plt.grid(True)  # Aggiunta della griglia
    plt.legend()  # Legenda per il grafico
    plt.savefig('plots/accuracy_vs_dropout.png')  # Salvataggio del grafico come immagine

    # ---- PLOT 2: Confronto tra Loss di Training e Test ----
    plt.figure(figsize=(8, 6))
    for model_type in ['Single Layer', 'Multi Layer']:
        model_results = results_df[results_df['model_type'] == model_type]
        plt.plot(model_results['dropout'], model_results['train_loss'], marker='o', linestyle='--', label=f'{model_type} Train Loss')
        plt.plot(model_results['dropout'], model_results['test_loss'], marker='o', linestyle='-', label=f'{model_type} Test Loss')

    plt.title('Training vs Validation Loss')  # Titolo del grafico
    plt.xlabel('Dropout Rate')  # Etichetta asse x (valori di dropout)
    plt.ylabel('Loss')  # Etichetta asse y (loss)
    plt.grid(True)  # Aggiunta della griglia
    plt.legend()  # Legenda per il grafico
    plt.savefig('plots/loss_vs_dropout.png')  # Salvataggio del grafico come immagine

    # ---- PLOT 3: Analisi dell'Overfitting (Training vs Test Accuracy) ----
    plt.figure(figsize=(8, 6))
    for model_type in ['Single Layer', 'Multi Layer']:
        model_results = results_df[results_df['model_type'] == model_type]
        plt.plot(model_results['dropout'], model_results['train_acc'], marker='o', linestyle='--', label=f'{model_type} Train Accuracy')
        plt.plot(model_results['dropout'], model_results['test_acc'], marker='o', linestyle='-', label=f'{model_type} Test Accuracy')

    plt.title('Overfitting Analysis: Train vs Test Accuracy')  # Titolo del grafico
    plt.xlabel('Dropout Rate')  # Etichetta asse x (valori di dropout)
    plt.ylabel('Accuracy')  # Etichetta asse y (accuratezza)
    plt.grid(True)  # Aggiunta della griglia
    plt.legend()  # Legenda per il grafico
    plt.savefig('plots/overfitting_analysis.png')  # Salvataggio del grafico come immagine

# Eseguiamo la funzione per generare i plot
plot_accuracy_loss()