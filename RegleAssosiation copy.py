import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules,fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# Fonction pour lire et préparer les données
def prepare_data(file_path):
    df = pd.read_csv(file_path)
 

    return df

# Fonction pour transformer les données en transactions
def transform_to_transactions(df):
    transactions = list(df["evenement"].apply(lambda x:x.split(",") ))
    # print(transactions)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df1 = pd.DataFrame(te_ary, columns=te.columns_)
    df1 = df1.replace(False,0)
    df1 = df1.replace(True,1)

    return df1

# Fonction pour extraire les règles d'association avec Apriori
def extract_rules_apriori(transactions, min_support=0.1, min_confidence=0.5, tw=4):
    # print("apriori ----")
    # print(transactions)
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    # frequent_itemsets = frequent_itemsets[frequent_itemsets['support'] >= tw]  # Application de 'tw'
    # print(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    # print("rules1",rules)
    return rules

# Fonction pour extraire les règles d'association avec FP-Growth
def extract_rules_fp_growth(transactions, min_support=0.1, min_confidence=0.5, tw=4):
    frequent_itemsets = fpgrowth(transactions, min_support=min_support, use_colnames=True)
    # frequent_itemsets = frequent_itemsets[frequent_itemsets['support'] >= tw]  # Application de 'tw'
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

# Analyse des règles d'association
def analyze_rules(rules, target_event):
    related_rules = rules[rules['consequents'].apply(lambda x: target_event in x)]
    return related_rules

# Main function
if __name__ == "__main__":
    file_path = 'events.csv'
    target_event = 'Average decrease of quantity'
    tw = 4  # Définir la valeur de 'tw'

    df = prepare_data(file_path)
    transactions = transform_to_transactions(df)

    # Extraction des règles avec Apriori
    apriori_rules = extract_rules_apriori(transactions, tw=tw)
    # print(apriori_rules)
    print("Apriori Rules:")
    print(analyze_rules(apriori_rules, target_event))

    # Extraction des règles avec FP-Growth
    fp_growth_rules = extract_rules_fp_growth(transactions, tw=tw)
    print(fp_growth_rules)

    print("FP-Growth Rules:")
    print(analyze_rules(fp_growth_rules, target_event))
