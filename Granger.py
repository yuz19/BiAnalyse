import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import mysql.connector
import json
import itertools
from datetime import datetime

class Granger:
    
    def __init__(self, columns , conn):
        self.conn = conn
        self.columns = columns

    def DynamiqueFormatDate(self, date_interval):
        cursor = self.conn.cursor()
        cursor.execute("SELECT date FROM time LIMIT 1")
        row = cursor.fetchone()

        if row:
            # Récupérer la valeur de la colonne de date
            date_value = str(row[0])
            date_interval = [datetime.strptime(date_str, "%d/%m/%Y") for date_str in date_interval]
            # Vérifier si la colonne de date contient le caractère /
            if '/' in date_value:
                date_components = date_value.split('/')
                if int(date_components[0]) > int(date_components[2]):
                    date_interval = [date_str.strftime("%Y/%m/%d") for date_str in date_interval]

            # Vérifier si la colonne de date contient le caractère :
            elif ':' in date_value:
                date_components = date_value.split(':')
                if int(date_components[0]) > int(date_components[2]):
                    date_interval = [date_str.strftime("%Y:%m:%d") for date_str in date_interval]
                else:
                    date_interval = [date_str.strftime("%Y:%m:%d") for date_str in date_interval]

            # Vérifier si la colonne de date contient le caractère -
            elif '-' in date_value:
                date_components = date_value.split('-')
                if int(date_components[0]) > int(date_components[2]):
                    date_interval = [date_str.strftime("%Y-%m-%d") for date_str in date_interval]
                else:
                    date_interval = [date_str.strftime("%d-%m-%Y") for date_str in date_interval]

        return date_interval

    def start(self,date_prefrence,date_interval):
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
        date_interval=self.DynamiqueFormatDate(date_interval)
        print(date_interval)
        for table_name, table_columns in tables_with_columns.items():
            for column in table_columns:
                cursor = self.conn.cursor()
                #static
                # cursor.execute(f"SELECT jour, mois, annee, {column} FROM {table_name},time where time.date_ID={table_name}.date_ID ORDER BY annee, mois,jour ASC  ")
                
                # cursor.execute(f"SELECT jour, mois, annee, SUM({column}) FROM {table_name},time WHERE time.date_ID={table_name}.date_ID GROUP BY annee, mois, jour ORDER BY annee, mois, jour ASC")
                if len(date_interval)==0:
                    cursor.execute(f"SELECT {date_prefrence}, SUM({column}) FROM {table_name},time WHERE time.date_ID={table_name}.date_ID GROUP BY {date_prefrence} ORDER BY {date_prefrence} ASC")
                else:
                    cursor.execute(f"SELECT {date_prefrence}, SUM({column}) FROM {table_name},time WHERE time.date_ID={table_name}.date_ID and time.date BETWEEN  '{date_interval[0]}' AND '{date_interval[1]}' GROUP BY {date_prefrence} ORDER BY {date_prefrence} ASC")

                rows = cursor.fetchall()
 
                if column in data_frames:
                    data_frames[column].extend([row[len(date_prefrence.split(','))] for row in rows])
                else:
                    data_frames[column] = [row[len(date_prefrence.split(','))] for row in rows]

                for  row in rows:
                    # datasave={
                    #     'index_JMA':f"{row[0]}-{row[1]}-{row[2]}" ,
                    #     'index_MA':f"{row[1]}-{row[2]}",
                    #     'index_A':f"{row[2]}",
                    #     'valeur':row[3]
                    # }
                    
                    if len(date_prefrence.split(','))==3:
                        indexdate=f"{row[0]}-{row[1]}-{row[2]}"
                    elif len(date_prefrence.split(','))==2:
                        indexdate=f"{row[0]}-{row[1]}"
                    else:
                        indexdate=f"{row[0]}"
                        
                    datasave={
                        'index_JMA': indexdate,
                        'valeur':row[len(date_prefrence.split(','))]
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
        results_all={}
        for col1, col2 in itertools.combinations(self.columns, 2):   

            # Perform the Granger causality test
            try:
                results= grangercausalitytests(df[[col1,col2]], max_lag, verbose=True)
            except Exception as e:
                print("Error during Granger causality test:")
                print(e)
                results= None

            test_F_values = []
            p_values = []
            affichage_granger = []
            lag_results=[]
            
            # Afficher et stocker les résultats dans les variables
            if results:
                for lag in range(1, max_lag + 1):
                    # print(f'\nRésultats pour le délai {lag}:')
                    test_F_value = results[lag][0]["ssr_ftest"][0]
                    p_value = results[lag][0]["ssr_ftest"][1]
                    # print(f'Test F : {test_F_value}')
                    # print(f'P-valeur : {p_value}')
                    
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

            # # Imprimer les résultats d'affichage
            # for affichage in affichage_granger:
            #     print(affichage)
                
            results_all[f"{col1}, {col2}"] = {
                "affichage_granger": affichage_granger,
                "lag_results": lag_results
            }
            

        # Convert the dictionary to a JSON string
        results_all = json.dumps(results_all)

        return results_all, self.columns, data
