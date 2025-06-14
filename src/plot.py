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
import matplotlib.ticker as mticker
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
    """
    device = config['device']
    train_loader, test_loader = get_loaders(batch_size=config['batch_size'])
    all_results = []
    
    for model_name, hidden_sizes in config['architectures'].items():
        for dropout_prob in tqdm(config['dropout_values'], desc=f"Testing {model_name}"):
            model = MLPFlexible(
                input_size=28*28,
                hidden_sizes=hidden_sizes,
                num_classes=10,
                dropout_prob=dropout_prob
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(config['epochs']):
                train_loss_epoch = train(model, train_loader, optimizer, criterion, device)
                test_acc, test_loss = test(model, test_loader, device)
                train_acc, _ = test(model, train_loader, device)
                
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
    confrontando i modelli. Le linee sono smussate con una media mobile.
    """
    print("Generazione grafici delle curve di apprendimento (solo linee smussate)...")
    dropout_to_compare = 0.4 
    markers = {"Training": "s", "Test": "o"}
    palette = "colorblind"
    window_size = 3

    for model_name in config['architectures'].keys():
        df_model = df[
            (df['model_name'] == model_name) &
            (df['dropout_prob'].isin([0.0, dropout_to_compare]))
        ].copy()

        metrics = ['train_acc', 'test_acc', 'train_loss', 'test_loss']
        for metric in metrics:
            df_model[f'{metric}_smooth'] = df_model.groupby('dropout_prob')[metric].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean()
            )
            
        df_acc = df_model.melt(
            id_vars=['epoch', 'dropout_prob'], 
            value_vars=['train_acc_smooth', 'test_acc_smooth'],
            var_name='Metric', 
            value_name='Accuratezza'
        )
        df_acc['Set'] = df_acc['Metric'].replace({'train_acc_smooth': 'Training', 'test_acc_smooth': 'Test'})

        df_loss = df_model.melt(
            id_vars=['epoch', 'dropout_prob'], 
            value_vars=['train_loss_smooth', 'test_loss_smooth'],
            var_name='Metric', 
            value_name='Loss'
        )
        df_loss['Set'] = df_loss['Metric'].replace({'train_loss_smooth': 'Training', 'test_loss_smooth': 'Test'})
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f'Curve di Apprendimento - {model_name}', fontsize=16)

        sns.lineplot(
            data=df_acc, x='epoch', y='Accuratezza', hue='dropout_prob', style='Set', 
            markers=markers, palette=palette, ax=axes[0], linewidth=2.5
        )
        axes[0].set_title('Accuratezza vs. Epoche')
        axes[0].set_xlabel('Epoca')
        axes[0].set_ylabel('Accuratezza')
        axes[0].grid(True, which='both', linestyle='--')
        axes[0].xaxis.set_major_locator(mticker.MultipleLocator(1))
        axes[0].legend(title='Dropout / Set')

        sns.lineplot(
            data=df_loss, x='epoch', y='Loss', hue='dropout_prob', style='Set', 
            markers=markers, palette=palette, ax=axes[1], linewidth=2.5
        )
        axes[1].set_title('Loss vs. Epoche')
        axes[1].set_xlabel('Epoca')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, which='both', linestyle='--')
        axes[1].xaxis.set_major_locator(mticker.MultipleLocator(1))
        axes[1].legend(title='Dropout / Set')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join('plots', f'learning_curves_smooth_only_{model_name.replace(" ", "_")}.png'))
        plt.close()

def plot_dropout_performance(df):
    """
    Genera e salva un grafico che mostra l'accuratezza finale sul test set
    al variare del tasso di dropout.
    """
    print("Generazione grafico performance vs. dropout...")
    final_results = df.loc[df.groupby(['model_name', 'dropout_prob'])['epoch'].idxmax()]
    
    plt.figure(figsize=(12, 7))
    palette = "colorblind"
    sns.lineplot(
        data=final_results, x='dropout_prob', y='test_acc', hue='model_name', 
        marker='o', palette=palette, linewidth=2
    )
    
    plt.title('Accuratezza di Test Finale vs. Tasso di Dropout', fontsize=16)
    plt.xlabel('Tasso di Dropout')
    plt.ylabel('Accuratezza di Test Finale')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(title='Architettura Modello')
    
    plt.savefig(os.path.join('plots', 'accuracy_vs_dropout.png'))
    plt.close()

# --- FUNZIONE MODIFICATA ---
def plot_overfitting_gap(df, config):
    """
    Genera grafici separati per ogni architettura, mostrando il gap di overfitting
    sull'accuratezza e confrontando il modello con e senza dropout.
    """
    print("Generazione grafici di analisi overfitting per accuratezza...")
    
    # Filtra solo per i dropout che vogliamo confrontare
    dropout_values_to_plot = [0.0, 0.4]
    df_plot = df[df['dropout_prob'].isin(dropout_values_to_plot)].copy()

    # Calcola il gap di accuratezza
    df_plot['accuracy_gap'] = df_plot['train_acc'] - df_plot['test_acc']

    # Applica la media mobile per smussare le linee del gap
    window_size = 3
    df_plot['accuracy_gap_smooth'] = df_plot.groupby(['model_name', 'dropout_prob'])['accuracy_gap'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )

    # Itera su ogni architettura per creare un grafico separato
    for model_name in config['architectures'].keys():
        df_model_specific = df_plot[df_plot['model_name'] == model_name]
        
        # Crea una nuova figura per ogni modello
        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        palette = "colorblind"

        sns.lineplot(
            data=df_model_specific,
            x='epoch',
            y='accuracy_gap_smooth',
            hue='dropout_prob',
            palette=palette,
            ax=ax,
            linewidth=2.5,
            marker='o',
            markersize=6,
        )
        
        ax.set_title(f'Gap di Accuratezza (Overfitting) - {model_name}')
        ax.set_xlabel('Epoca')
        ax.set_ylabel('Differenza Accuratezza (Train - Test)')
        ax.grid(True, which='both', linestyle='--')
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.legend(title='Tasso di Dropout')
        ax.axhline(0, color='k', linestyle='--', alpha=0.7)

        # Salva il grafico con un nome specifico per il modello
        filename_safe_model_name = model_name.replace(" ", "_")
        plt.tight_layout()
        plt.savefig(os.path.join('plots', f'overfitting_gap_accuracy_{filename_safe_model_name}.png'))
        plt.close()


def main():
    """
    Funzione principale che orchestra l'esecuzione degli esperimenti e la generazione dei grafici.
    """
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 20,
        'architectures': {
            'MLP 1 Strato Nascosto': [512],
            'MLP Multi-Strato': [512, 256, 128]
        },
        'dropout_values': [0.0, 0.2, 0.4, 0.6]
    }
    
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
        
    sns.set_theme(style="whitegrid")
    
    plot_learning_curves(results_df, config)
    plot_dropout_performance(results_df)
    plot_overfitting_gap(results_df, config)
    
    print("\nGrafici generati e salvati nella cartella 'plots'.")


if __name__ == '__main__':
    main()