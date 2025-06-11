# File per l'addestramento, la valutazione e la visualizzazione delle performance dei modelli.
# Questo script esegue gli esperimenti, salva i risultati su un file CSV
# e genera i grafici richiesti per l'analisi.

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# Importa le funzioni e le classi necessarie dai file del progetto
from data_loader import get_loaders
from model import MLPFlexible
from train import train
from test import test

def run_experiments(config):
    """
    Esegue gli esperimenti di training e test per diverse configurazioni di modello
    e salva i risultati storici (per epoca) in un DataFrame pandas.

    Args:
        config (dict): Un dizionario contenente gli iperparametri e le configurazioni.

    Returns:
        pd.DataFrame: Un DataFrame contenente i risultati dettagliati di tutti gli esperimenti.
    """
    device = config['device']
    
    # Carica i dati una sola volta
    train_loader, test_loader = get_loaders(batch_size=config['batch_size'])
    
    # Lista per conservare tutti i risultati
    all_results = []
    
    # Cicla su ogni architettura di modello definita nella configurazione
    for model_name, hidden_sizes in config['architectures'].items():
        # Cicla su ogni valore di dropout da testare
        for dropout_prob in tqdm(config['dropout_values'], desc=f"Testing {model_name}"):
            
            # Inizializza il modello e l'ottimizzatore
            model = MLPFlexible(
                input_size=28*28,
                hidden_sizes=hidden_sizes,
                num_classes=10,
                dropout_prob=dropout_prob
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            # Cicla per il numero di epoche
            for epoch in range(config['epochs']):
                # Esegui un'epoca di training
                train_loss_epoch = train(model, train_loader, optimizer, criterion, device)
                
                # Valuta il modello sul test set e sul training set (per avere l'accuratezza)
                test_acc, test_loss = test(model, test_loader, device)
                train_acc, _ = test(model, train_loader, device) # Calcoliamo l'accuratezza di training usando la funzione di test
                
                # Salva i risultati di questa epoca
                all_results.append({
                    'model_name': model_name,
                    'hidden_sizes': str(hidden_sizes),
                    'dropout_prob': dropout_prob,
                    'epoch': epoch + 1,
                    'train_loss': train_loss_epoch,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc
                })
                
    return pd.DataFrame(all_results)

def plot_learning_curves(df, config):
    """
    Genera e salva i grafici delle curve di apprendimento (accuratezza e loss)
    confrontando i modelli con e senza dropout.
    """
    print("Generazione grafici delle curve di apprendimento...")
    # Confrontiamo il modello senza dropout (0.0) con un modello con dropout (es. 0.4)
    dropout_to_compare = 0.4 
    
    for model_name in config['architectures'].keys():
        # Filtra i dati per il modello corrente e per i valori di dropout scelti
        df_plot = df[
            (df['model_name'] == model_name) &
            (df['dropout_prob'].isin([0.0, dropout_to_compare]))
        ]
        
        # Crea una figura con due subplot (uno per l'accuratezza, uno per la loss)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Curve di Apprendimento - {model_name}', fontsize=16)

        # Plot per l'Accuratezza
        sns.lineplot(data=df_plot, x='epoch', y='train_acc', hue='dropout_prob', style='dropout_prob', ax=axes[0], palette='viridis', legend=False, dashes=[(1,0), (2,2)])
        sns.lineplot(data=df_plot, x='epoch', y='test_acc', hue='dropout_prob', style='dropout_prob', ax=axes[0], palette='viridis')
        axes[0].set_title('Accuratezza vs. Epoche')
        axes[0].set_xlabel('Epoca')
        axes[0].set_ylabel('Accuratezza')
        axes[0].grid(True)
        # Crea legenda manualmente per chiarezza
        handles, _ = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, [f'Test (dropout={p})' for p in [0.0, dropout_to_compare]] + [f'Train (dropout={p})' for p in [0.0, dropout_to_compare]], title='Dropout')


        # Plot per la Loss
        sns.lineplot(data=df_plot, x='epoch', y='train_loss', hue='dropout_prob', style='dropout_prob', ax=axes[1], palette='viridis', legend=False, dashes=[(1,0), (2,2)])
        sns.lineplot(data=df_plot, x='epoch', y='test_loss', hue='dropout_prob', style='dropout_prob', ax=axes[1], palette='viridis')
        axes[1].set_title('Loss vs. Epoche')
        axes[1].set_xlabel('Epoca')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
        handles, _ = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, [f'Test (dropout={p})' for p in [0.0, dropout_to_compare]] + [f'Train (dropout={p})' for p in [0.0, dropout_to_compare]], title='Dropout')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join('plots', f'learning_curves_{model_name.replace(" ", "_")}.png'))
        plt.close()

def plot_dropout_performance(df):
    """
    Genera e salva un grafico che mostra l'accuratezza finale sul test set
    al variare del tasso di dropout.
    """
    print("Generazione grafico performance vs. dropout...")
    # Estrai i risultati dell'ultima epoca per ogni esperimento
    final_results = df.loc[df.groupby(['model_name', 'dropout_prob'])['epoch'].idxmax()]
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=final_results, x='dropout_prob', y='test_acc', hue='model_name', marker='o')
    
    plt.title('Accuratezza di Test vs. Tasso di Dropout', fontsize=16)
    plt.xlabel('Tasso di Dropout')
    plt.ylabel('Accuratezza di Test Finale')
    plt.grid(True)
    plt.legend(title='Architettura Modello')
    
    plt.savefig(os.path.join('plots', 'accuracy_vs_dropout.png'))
    plt.close()


def main():
    """
    Funzione principale che orchestra l'esecuzione degli esperimenti e la generazione dei grafici.
    """
    # --- CONFIGURAZIONE DEGLI ESPERIMENTI ---
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 20, # Un numero di epoche sufficiente a vedere l'overfitting
        'architectures': {
            'MLP 1 Strato Nascosto': [512],
            'MLP Multi-Strato': [512, 256, 128]
        },
        'dropout_values': [0.0, 0.2, 0.4, 0.6]
    }
    
    # Crea la cartella dei plot e dei risultati se non esistono
    os.makedirs('plots', exist_ok=True)
    results_file = 'results.csv'
    
    if os.path.exists(results_file):
        print(f"Trovato file '{results_file}'. Caricamento dati esistenti...")
        results_df = pd.read_csv(results_file)
    else:
        print("Nessun file di risultati trovato. Esecuzione esperimenti...")
        results_df = run_experiments(config)
        results_df.to_csv(results_file, index=False)
        print(f"Esperimenti completati e risultati salvati in '{results_file}'.")
        
    # Applica uno stile grafico piacevole
    sns.set_theme(style="whitegrid")
    
    # Genera e salva i grafici
    plot_learning_curves(results_df, config)
    plot_dropout_performance(results_df)
    
    print("\nGrafici generati e salvati nella cartella 'plots'.")


if __name__ == '__main__':
    main()