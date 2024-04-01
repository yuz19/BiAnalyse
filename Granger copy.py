import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import mysql.connector

class Granger:
    
    def __init__(self, columns , conn):
        self.conn = conn
        self.columns = columns
 
    
    def start(self):
        tables_with_columns = {}
        # Récupérer les tables associées à chaque colonne spécifiée
        for column in self.columns:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME = '{column}'")
            rows = cursor.fetchall()
            for row in rows:
                table_name = row[0]
                if table_name in tables_with_columns:
                    tables_with_columns[table_name].append(column)
                else:
                    tables_with_columns[table_name] = [column]

        if not tables_with_columns:
            return {"message": "No tables found containing the specified columns."}

        # Récupérer les données pour chaque colonne et les stocker dans un DataFrame
        data_frames = {}
        data={}
        
        for table_name, table_columns in tables_with_columns.items():
            for column in table_columns:
                cursor = self.conn.cursor()
                #static
                # cursor.execute(f"SELECT jour, mois, annee, {column} FROM {table_name},time where time.date_ID={table_name}.date_ID ORDER BY annee, mois,jour ASC  ")
                cursor.execute(f"SELECT jour, mois, annee, SUM({column}) FROM {table_name},time WHERE time.date_ID={table_name}.date_ID GROUP BY annee, mois, jour ORDER BY annee, mois, jour ASC")

                rows = cursor.fetchall()
                
                if column in data_frames:
                    data_frames[column].extend([row[3] for row in rows])
                else:
                    data_frames[column] = [row[3] for row in rows]

                for  row in rows:
                    datasave={
                        'index_JMA':f"{row[0]}-{row[1]}-{row[2]}" ,
                        'index_MA':f"{row[1]}-{row[2]}",
                        'index_A':f"{row[2]}",
                        'valeur':row[3]
                    }
                    if column in data:
                            data[column].append(datasave)
                    else:
                        data[column] = [datasave]

        # print("len",len(data[column]))        
        # Create a DataFrame from the dictionary
        df = pd.DataFrame(data_frames)
        # Print the shape of the DataFrame
        print("Shape of DataFrame:")
        print(df.shape)
        

        # Replace missing values with the mean of each column
        df = df.fillna(df.mean())
        # Effectuer le test de causalité de Granger pour chaque colonne dans cette table
        max_lag = 5  # Choisissez le nombre maximal de retards à tester
        print(df)

        # Perform the Granger causality test
        try:
            results = grangercausalitytests(df, max_lag, verbose=True)
        except Exception as e:
            print("Error during Granger causality test:")
            print(e)
            results = None
        test_F_values = []
        p_values = []
        affichage_granger = []
        lag_results=[]
        # Afficher et stocker les résultats dans les variables
        if results:
            for lag in range(1, max_lag + 1):
                print(f'\nRésultats pour le délai {lag}:')
                test_F_value = results[lag][0]["ssr_ftest"][0]
                p_value = results[lag][0]["ssr_ftest"][1]
                print(f'Test F : {test_F_value}')
                print(f'P-valeur : {p_value}')
                
                lag_results.append({
                'lag': lag,
                'test_F_value': test_F_value,
                'p_value': p_value
                })
                
                # Stocker les résultats dans les listes
                test_F_values.append(test_F_value)
                p_values.append(p_value)

        # Vérification de la causalité
        significant_lags = [lag for lag, p_value in enumerate(p_values, 1) if p_value < 0.05]

        if significant_lags:
            affichage_granger.append(f'Causalité trouvée pour au moins un délai : {significant_lags}')
        else:
            affichage_granger.append('Aucune causalité trouvée pour tous les délais testés.')

        # Imprimer les résultats d'affichage
        for affichage in affichage_granger:
            print(affichage)
  

        
        return affichage_granger,self.columns,data,lag_results

