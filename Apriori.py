def custom_apriori(columns):
    tables_with_columns = {}

    # Récupérer les tables associées à chaque colonne spécifiée
    for column in columns:
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME = '{column}'")
        rows = cursor.fetchall()
        for row in rows:
            table_name = row[0]
            if table_name in tables_with_columns:
                tables_with_columns[table_name].append(column)
            else:
                tables_with_columns[table_name] = [column]

    if not tables_with_columns:
        return Response({"message": "No tables found containing the specified columns."})

    # Récupérer les données pour chaque colonne et les stocker dans un DataFrame
    data_frames = {}
    for table_name, table_columns in tables_with_columns.items():
        for column in table_columns:
            cursor = conn.cursor()
            cursor.execute(f"SELECT {column} FROM {table_name}")
            rows = cursor.fetchall()
            if column in data_frames:
                data_frames[column].extend([row[0] for row in rows])
            else:
                data_frames[column] = [row[0] for row in rows]

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data_frames)

    # Remplacer les valeurs manquantes avec la moyenne de chaque colonne
    df = df.fillna(df.mean())

    # Identifier les colonnes nécessitant une conversion en chaînes
    columns_to_str = df.columns.tolist()

    # Convertir uniquement les colonnes nécessaires en chaînes
    df[columns_to_str] = df[columns_to_str].astype(str)

    # Convertir les données en transactions
    transactions = df.values.tolist()

    # Appliquer l'algorithme Apriori
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=False)

    # Afficher les ensembles fréquents dans le terminal
    print("\nEnsembles fréquents :")
    print(frequent_itemsets)

    
    # html_content = ""
    return frequent_itemsets
