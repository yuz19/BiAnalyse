import mysql.connector
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from Proposer import Proposer
import itertools

try:
    # Establish connection to the database
    conn = mysql.connector.connect(
        host="localhost",
        database="ventes_enligne",
        user="root",
        password="1962",
        port="3306"  # Changed to the default MySQL port
    )
except mysql.connector.Error as err:
    print(f"Error: {err}")
    exit(1)

cursor = conn.cursor()


def granger(columns,cursor):
        # Initialize data_frames dictionary
    data_frames = {}
    for column in columns:

        cursor.execute(f"SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME = '{column}'")
        row = cursor.fetchone()
        table_name = row[0] if row else None

        if table_name:
            cursor.execute(f"""
                SELECT annee, mois, jour, SUM({column}) 
                FROM {table_name}, time 
                WHERE time.date_ID = {table_name}.date_ID 
                GROUP BY annee, mois, jour 
                ORDER BY annee, mois, jour ASC
            """)
            rows = cursor.fetchall()

            if rows:
                date_preference_static = "annee, mois, jour"
                if column in data_frames:
                    data_frames[column].extend([row[len(date_preference_static.split(','))] for row in rows])
                else:
                    data_frames[column] = [row[len(date_preference_static.split(','))] for row in rows]

    # Create DataFrame from the collected data
    df = pd.DataFrame(data_frames)
    print("Shape of DataFrame:")
    print(df.shape)
    # Replace missing values with the mean of each column
    df = df.fillna(df.mean())
    # Effectuer le test de causalité de Granger pour chaque colonne dans cette table
    max_lag = 5  # Choisissez le nombre maximal de retards à tester
    results_all=[]
    error=""
    
    for col1, col2 in itertools.combinations(columns, 2):   
            
        # Perform the Granger causality test
        try:
            results= grangercausalitytests(df[[col1,col2]], max_lag, verbose=True)
        except ValueError as e:
            # Check if the exception message contains "Insufficient observations."
            if "Insufficient observations." in str(e):
                # Handle the case of insufficient observations
                error=" (Insufficient Data )"
                results= None

            else:
                # Handle other ValueError cases
                print("Other ValueError occurred:", e)
        except Exception as e:
            # Handle other types of exceptions
            print("An unexpected error occurred:", e)
        

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
            min_p_value_index = p_values.index(min(p_values))
            min_test_F_value = test_F_values[min_p_value_index]
            affichage_granger.append(f'Causalité trouvée pour au moins un délai : {significant_lags}')
        else:
            affichage_granger.append(f'Aucune causalité trouvée pour tous les délais testés.{error}')

        # # Imprimer les résultats d'affichage
        # for affichage in affichage_granger:
        #     print(affichage)
            
        results_all.append({
            "columns": [col1, col2],
            "affichage_granger": affichage_granger,
            "lag_results": lag_results
        })
            
        # Vérification de la causalité
        significant_lags = [lag for lag, p_value in enumerate(p_values, 1) if p_value < 0.05]
        min_p_value=1
        if significant_lags:
            affichage_granger.append(f'Causalité trouvée pour au moins un délai : {significant_lags}')
            # Print the minimum p-value
            if p_values:
                min_p_value = round(min(p_values), 3)
                print(f"{col1},{col2} The minimum p-value is: {min_p_value}, associated F-value is: {min_test_F_value}")
        else:
            affichage_granger.append(f'Aucune causalité trouvée pour tous les délais testés.{error}')


granger(["prix_ventes","reduction"],cursor)     
# Close the cursor and the connection
cursor.close()
conn.close()

print("end")