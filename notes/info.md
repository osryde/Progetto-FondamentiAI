# Progetto di Fondamenti dell'Intelligenza Artificiale
---

**Obiettivo**: Studiare il dropout come tecnica di regolarizzazione per le reti neurali. Implementare un MLP a uno strato nascosto e a più strati nascosti, confrontando le prestazioni con e senza dropout su un dataset di classificazione (ad esempio MNIST o FashionMNIST).

Ma cos'è il **dropout**? 
Il dropout è una tecnica di regolarizzazione usata nelle reti neurali per ridurre l'overfitting. Con **overfitting** si indica il caso in cui il modello diventa troppo dipendente dai dati di training, rendendolo inefficiente nella capacità di generalizzare sui dati nuovi.

In poche parole durante l'addestramento, il **dropout** disattiva in modo casuale (cioè azzera) una percentuale di neuroni in un layer. Questo impedisce alla rete di diventare troppo dipendente da alcuni neuroni specifici.

- Ad esempio, con dropout(p=0.5), il 50% dei neuroni viene ignorato a ogni iterazione.
- Durante il test (inference, cioè fase di test su dati nuovi), tutti i neuroni sono attivi, ma i pesi vengono scalati per compensare.

>Durante il training, il dropout aumenta l’output dei neuroni attivi moltiplicandolo per 1 / (1 - p) per mantenere costante l’output medio.
Durante il test (inference), tutti i neuroni sono attivi e non serve alcuna modifica. Parliamo dunque di [**inverted dropout**](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9/).

Quindi durante la fase di dropout avremo un incremento del valore dei neuroni attivi, in modo da rendere l'output medio delle ue fasi (training e test) pressochè simili.

## Osservazioni
---
- Il dataset utilizzato (**MNIST**) è già presente in <code>torchvision.datasets</code>

## Slides
---

Idee per slides:
- Titolo
- Background: Cos'è un MLP tradizionale e come viene addestrato (2 slide)
- Teorema 1 strato nascosto e perchè nonostante ciò vengono utilizzate reti con più strati nascosti
- Cos'è il Dropout e perchè è utile (2/3 slide)
- Una slide dove mostro l'invocazione dell'MLP flessibile con i parametri ad esso legati (1 slide + 1 slide di esempio)
- Il dataset utilizzato per addestramento e test della rete (1 slide + 1 slide con immagini d'esempio + 1 slide su come questo tipo di dato viene gestito)
- Risultati (n-slide dove vengono mostrati i risultati di benchmark su MNIST)
- Considerazioni finali: Parlare dei risultati e come altre tipologie di dati gioverebbero ulteriormente del dropout (o forse no)

In totale devono venire fuori circa 10/15 minuti di presentazione.