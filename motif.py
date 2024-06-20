import pandas as pd

# Fonction pour compter le support des itemsets
def count_support(transactions, itemsets):
    support_counts = {}
    for transaction in transactions:
        for itemset in itemsets:
            if set(itemset).issubset(set(transaction)):
                support_counts.setdefault(tuple(itemset), 0)
                support_counts[tuple(itemset)] += 1
    return support_counts

# Fonction pour générer les itemsets candidats de taille k+1 à partir des itemsets de taille k
def generate_candidate_itemsets(frequent_itemsets):
    candidate_itemsets = []
    for i in range(len(frequent_itemsets)):
        for j in range(i+1, len(frequent_itemsets)):
            # Fusionner les itemsets s'ils ont les mêmes (k-1) premiers éléments
            if frequent_itemsets[i][:-1] == frequent_itemsets[j][:-1]:
                # Concaténer les deux derniers éléments pour former un nouveau tuple
                new_itemset = tuple(list(frequent_itemsets[i]) + [frequent_itemsets[j][-1]])
                candidate_itemsets.append(new_itemset)
    return candidate_itemsets

# Fonction pour évaluer la séquentialité d'un motif
def evaluate_sequentiality(transactions, motif):
    count_sequence = 0
    count_occurrences = 0
    for transaction in transactions:
        if all(event in transaction for event in motif):
            count_occurrences += 1
            if transaction.index(motif[0]) < transaction.index(motif[1]):
                count_sequence += 1
    sequentiality_ratio = count_sequence / count_occurrences if count_occurrences > 0 else 0
    return sequentiality_ratio

# Fonction principale pour l'algorithme AprioriAll
def apriori_all(transactions, min_support):
    # Initialiser les itemsets de taille 1 (singletons)
    singletons = [[item] for sublist in transactions for item in sublist]
    frequent_itemsets = [itemset for itemset, count in count_support(transactions, singletons).items() if count >= min_support]

    # Tant qu'il y a de nouveaux itemsets fréquents à générer
    while frequent_itemsets:
        print("Frequent itemsets of length", len(frequent_itemsets[0]), ":")
        for itemset in frequent_itemsets:
            print(itemset)

        # Générer les candidats de taille k+1 à partir des itemsets de taille k
        candidate_itemsets = generate_candidate_itemsets(frequent_itemsets)
        
        # Compter le support des candidats
        support_counts = count_support(transactions, candidate_itemsets)
        
        # Filtrer les candidats qui ont un support suffisant
        frequent_itemsets = [list(itemset) for itemset, count in support_counts.items() if count >= min_support]

        # Évaluation de la séquentialité pour chaque motif fréquent
        for motif in frequent_itemsets:
            sequentiality_ratio = evaluate_sequentiality(transactions, motif)
            # print("---------Sequentiality ratio for", motif, ":", sequentiality_ratio)
            # Vérifier si l'événement cible est suivi séquentiellement par l'événement cause dans les motifs fréquents avec un ratio de séquentialité de 1.0
            target_event = 'Weak decrease-quantity'
            cause_event = 'Important increase-retour_quantity'
            if target_event in motif and cause_event in motif and sequentiality_ratio == 1.0:
                print(f"L'événement cible '{target_event}' est suivi séquentiellement par l'événement cause '{cause_event}' dans le motif {motif} avec un ratio de séquentialité de 1.0.")
                return


# Charger les données d'événements depuis un fichier CSV
events_data = pd.read_csv('events.csv')['evenement'].apply(lambda x: x.split(', ')).tolist()

# Paramètres
min_support = 2

# Exécuter l'algorithme AprioriAll
apriori_all(events_data, min_support)
