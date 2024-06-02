import mysql.connector
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from Proposer import Proposer

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

# Initialize data_frames dictionary
data_frames = {}

# Query and process the 'profit' column
column = "profit"
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

# Query and process the 'retour_quantity' column
column = "retour_quantity"
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

# Fill NaN values with the mean of the column
df = df.fillna(df.mean())

# Define columns for the Granger causality test
col1 = "profit"
col2 = "retour_quantity"

max_lag = 5  # Choose the maximum number of lags to test
results_all = []
error = ""

# Perform the Granger causality test
try:
    results = grangercausalitytests(df[[col1, col2]], max_lag, verbose=True)
except ValueError as e:
    # Check if the exception message contains "Insufficient observations."
    if "Insufficient observations." in str(e):
        # Handle the case of insufficient observations
        error = " (Insufficient Data )"
        results = None
    else:
        # Handle other ValueError cases
        print("Other ValueError occurred:", e)
        results = None
except Exception as e:
    # Handle other types of exceptions
    print("An unexpected error occurred:", e)
    results = None
    
test_F_values = []
p_values = []
affichage_granger = []
lag_results=[]

# Afficher et stocker les résultats dans les variables
if results:
    for lag in range(1, max_lag + 1):
        
        test_F_value = results[lag][0]["ssr_ftest"][0]
        p_value = results[lag][0]["ssr_ftest"][1]
        
        lag_results.append({
        'lag': lag,
        'test_F_value': test_F_value,
        'p_value': p_value
        })
 
        test_F_values.append(test_F_value)
        p_values.append(p_value)

# Vérification de la causalité
significant_lags = [lag for lag, p_value in enumerate(p_values, 1) if p_value < 0.05]
min_p_value=1
if significant_lags:
    affichage_granger.append(f'Causalité trouvée pour au moins un délai : {significant_lags}')
    # Print the minimum p-value
    if p_values:
        min_p_value = round(min(p_values), 3)
        print(f"The minimum p-value is: {min_p_value}")
else:
    affichage_granger.append(f'Aucune causalité trouvée pour tous les délais testés.{error}')
 
# #################Prposer
import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime
import math
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
import itertools
from CalculeCauslite import CalculeCauslite
from CausaleDegree import CausaleDegree
 
 
class event:
    
    def __init__(self,ID_e , Measure, pos_dates,RefEvent):
        self.ID_e=ID_e
        self.Measure = Measure
        self.pos_dates = pos_dates
        self.RefEvent=RefEvent
 

def training_alpha(data):
    # Diviser les données en ensembles d'entraînement et de test
    train_size = int(len(data) * 0.8)  # 80% pour l'entraînement
    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]
    
    # Définition de la liste des valeurs de alpha à tester
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]

    # Initialisation des variables pour stocker le meilleur modèle et son erreur
    best_alpha = None
    best_mse = float('inf')

    # Validation croisée pour choisir la meilleure valeur de alpha
    for alpha in alphas:
        # Entraînement du modèle SES avec la valeur de alpha courante
        model = SimpleExpSmoothing(train_data).fit(smoothing_level=alpha)
        
        # Prédiction sur l'ensemble de test
        predictions = model.forecast(len(test_data))
        
        # Calcul de l'erreur quadratique moyenne (MSE)
        mse = mean_squared_error(test_data, predictions)
        
        # Mise à jour du meilleur modèle si nécessaire
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    return best_alpha

def save_peaks( values):
    peaks = {}
    peaks[0] = {"index": 0, "value": values[0]}
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            peaks[i] = {"index": i, "value": values[i]}
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            peaks[i] = {"index": i, "value": values[i]}
    return peaks
# Tendance + amplitude +date_debut +date_fin
def analyze_tend_intervals( peaks,df,TDate):
    tend_intervals = []
    peak_indices = list(peaks.keys())
    index = 1  # Start index at 1

    if len(peak_indices) < 2:
        return tend_intervals  # Return an empty list if there are less than 2 peaks

    for i in range(1, len(peak_indices)):
        prev_peak_index = peak_indices[i - 1]
        curr_peak_index = peak_indices[i]

        prev_peak_value = peaks[prev_peak_index]["value"]
        curr_peak_value = peaks[curr_peak_index]["value"]
        
        prev_peak_index = int(prev_peak_index)
        curr_peak_index = int(curr_peak_index)
        prev_peak_value = int(prev_peak_value)
        curr_peak_value = int(curr_peak_value)
        
        trend_type = "static" if prev_peak_value == curr_peak_value else ("augmentation" if prev_peak_value < curr_peak_value else "diminution")
    
        # print(index_JMA_value
        interval = {
            f"tendance {index}": {
                "type": trend_type,
                "interval": [
                    {
                        "date-debut": df.iloc[prev_peak_index] ['index_JMA'],
                        "value": prev_peak_value
                    },
                    {
                        "date-fin": df.iloc[curr_peak_index]['index_JMA'],
                        "value": curr_peak_value
                    }
                ],
                "amplitude": curr_peak_value - prev_peak_value
            }
        }
        # print(interval)
        tend_intervals.append(interval)
        index += 1  # Increment index for the next trend

    return tend_intervals

def qualification( tend_intervals):
    
    max_amplitude = max([abs(interval[f"tendance {i}"]["amplitude"]) for i, interval in enumerate(tend_intervals, 1)])
    min_amplitude = min([abs(interval[f"tendance {i}"]["amplitude"]) for i, interval in enumerate(tend_intervals, 1)])

    for i, interval in enumerate(tend_intervals, 1):
        amplitude = abs(interval[f"tendance {i}"]["amplitude"])
        if amplitude >= max_amplitude * (2/3):
            interval[f"tendance {i}"]["qualification"] = "Important"
        elif max_amplitude * (1/3) <= amplitude < max_amplitude * (2/3):
            interval[f"tendance {i}"]["qualification"] = "Average"
        else:
            interval[f"tendance {i}"]["qualification"] = "Weak"

        
    return tend_intervals

def Evenement( tend_intervals,index):
    
    Evenement_array = []
    for i, interval in enumerate(tend_intervals, 1):
        if interval[f"tendance {i}"]["type"] !="static":
            qualif = interval[f"tendance {i}"]["qualification"]
            peak_type =  "High" if interval[f"tendance {i}"]["type"] == 'diminution' else "Low"
            # peak_type =  "Low" if interval[f"tendance {i}"]["type"] == 'augmentation' else "High"
        
            status = "decrease" if interval[f"tendance {i}"]["type"] == 'diminution' else "increase"
            date_fin = interval[f"tendance {i}"]['interval'][1]['date-fin']  # Accéder à la clé 'date-fin' du deuxième élément de la liste 'interval'
            # date_debut = interval[f"tendance {i}"]['interval'][0]['date-debut']  # Accéder à la clé 'date-fin' du deuxième élément de la liste 'interval'
            
            max_value = max([abs(value["value"]) for i, value in enumerate(interval[f"tendance {i}"]['interval'], 1)])
            min_value = min([abs(value["value"]) for i, value in enumerate(interval[f"tendance {i}"]['interval'], 1)])
            

            
            Evenement = {
                "Evenement": f"{peak_type} peak of {qualif} {status}",
                "Ref":ref_evenement(f"{peak_type} peak of {qualif} {status}",index),
                "Date": date_fin,
                "Optimum":min_value if interval[f"tendance {i}"]["type"] == 'diminution' else max_value
            }
            Evenement_array.append(Evenement)
        
    return Evenement_array

def ref_evenement( Evenement, index):
    switch = {
        "Low peak of Weak increase": f"e{index}_1",
        "Low peak of Average increase": f"e{index}_2",
        "Low peak of Important increase": f"e{index}_3",
        
        "High peak of Weak decrease": f"e{index}_4",
        "High peak of Average decrease": f"e{index}_5",
        "High peak of Important decrease": f"e{index}_6"
    }
    
    return switch.get(Evenement, "Invalid event")


def CausaleD( evenement_all, columns):
    DI_allEvenet = []
    for idx1, col1 in enumerate(columns):
        # print (idx1)
        # print(col1)
        for idx2, col2 in enumerate(columns):
            if idx1 < idx2:  # Pour éviter de traiter les paires de colonnes deux fois
                event_DI1 = []
                for i in range(1, 13):
                    if f"e{idx1}_{i}"in evenement_all[col1]:
                        for j in range(1, 13):
                            if f"e{idx2}_{j}" in evenement_all[col2]:
                                e1_cause = event(col1, evenement_all[col1][f"e{idx1}_{i}"], RefEvent=f"e{idx1}_{i}")
                                e2_effect = event(col2, evenement_all[col2][f"e{idx2}_{j}"], RefEvent=f"e{idx2}_{j}")
                                CausaleDegree_instance = CausaleDegree()
                                di_test = CausaleDegree_instance.DI_causal_2(e1_cause, e2_effect)
                                # print(col1, ',', i, '-', col2, ',', j, ":", di_test)
                                event_DI1.append((f"e{idx1}_{i}-e{idx2}_{j}", di_test))
                event_DI2 = []
                for i in range(1, 13):
                    if f"e{idx2}_{i}" in evenement_all[col2]:
                        for j in range(1, 13):
                            if f"e{idx1}_{j}" in evenement_all[col1]:
                                e2_effect = event(col1, evenement_all[col1][f"e{idx1}_{j}"], RefEvent=f"e{idx1}_{j}")
                                e1_cause = event(col2, evenement_all[col2][f"e{idx2}_{i}"], RefEvent=f"e{idx2}_{i}")
                                CausaleDegree_instance = CausaleDegree()
                                di_test = CausaleDegree_instance.DI_causal_2(e1_cause, e2_effect)
                                # print(col2, ',', i, '-', col1, ',', j, ":", di_test)
                                event_DI2.append((f"e{idx2}_{i}-e{idx1}_{j}", di_test))

                DI_allEvenet.append({
                    "columns": [col1, col2],
                    "evenet_DI sens1": event_DI1,
                    "evenet_DI sens2": event_DI2
                })

    return DI_allEvenet

    
def CalculeCausa(evenement_all,columns):
    E_array = []

    # print(evenement_all)
    CalculeCauslite_instance=CalculeCauslite()
    for index, col in enumerate(columns): 

        switch = {
            f"e{index}_1": "Low peak of Weak increase",

            f"e{index}_2": "Low peak of Average increase",

            f"e{index}_3": "Low peak of Important increase",

            f"e{index}_4": "High peak of Weak decrease",

            f"e{index}_5": "High peak of Average decrease",

            f"e{index}_6": "High peak of Important decrease"
        }

    
        for i in range(1, 13):        
            if f"e{index}_{i}" in evenement_all[col]:
                print("list")
                print(f"e{index}_{i}")
                ID_e= switch.get(f"e{index}_{i}", "Invalid event").split("of")[1]+"-"+col
                # print("dates",col,":",f"e{index}_{i}","", evenement_all[col][f"e{index}_{i}"])
                E = event(ID_e,col, evenement_all[col][f"e{index}_{i}"], RefEvent=f"e{index}_{i}")
                E_array.append(E)


    # # print("Event externe")
    # for index,evenetE in evenement_all['externe'].items():
    #     E = event(index,"externe", evenetE, RefEvent=index)
    #     # print(E.ID_e)  
    #     E_array.append(E)
        
    matrice,array_Causes=CalculeCauslite_instance.creation_matrice_influence(E_array)
    # print(matrice)
    return matrice,array_Causes   
 

results_all=[]
results=[]
data_all={}
evenement_all={}
dimension_all={}
TDate="index_JMA"
 
# print(date_interval)
columns=["profit","retour_quantity"]
for index, column in enumerate(columns):
    cursor.execute(f"SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME = '{column}'")
    row = cursor.fetchall()
    # print(row)
    table_name = row[0][0]
    # print("table name", table_name)
 
    
    data_frames=[]

    cursor.execute(f"SELECT {date_preference_static}, SUM({column}) FROM {table_name},time WHERE time.date_ID={table_name}.date_ID GROUP BY {date_preference_static} ORDER BY {date_preference_static} ASC")
 

    rows = cursor.fetchall()
    
    # data_frames = [row[3] for row in rows]
    
    data = []

    for row in rows:
    
        if len(date_preference_static.split(','))==3:
            indexdate=f"{row[0]}-{row[1]}-{row[2]}"
        elif len(date_preference_static.split(','))==2:
            indexdate=f"{row[0]}-{row[1]}"
        else:
            indexdate=f"{row[0]}"
            
        datasave={
            'index_JMA': indexdate,
            'valeur':row[len(date_preference_static.split(','))]
        }
        # datasave = {
        #     'index_JMA': f"{row[0]}-{row[1]}-{row[2]}",
        #     'index_MA': f"{row[1]}-{row[2]}",
        #     'index_A': f"{row[2]}",
        #     'valeur': row[3]
        # }
        data.append(datasave)
        

 

    # print("data",data)
    # Créer un DataFrame à partir des données
    # df = pd.DataFrame(data_frames,columns=[column])
    # df[column] = df[column].astype(float)
    # # print(data)
    df = pd.DataFrame(data)
    # df['valeur'] = df['valeur'].astype(float)
    df['valeur'] = pd.to_numeric(df['valeur'], errors='coerce')
    # print(df)
    # Détecter les points hauts et bas
    # values = df[column].values
    # Perform smoothing on the data
    # alpha = 0.2  # Smoothing parameter
    data_smoothed = SimpleExpSmoothing(df['valeur']).fit(smoothing_level=training_alpha(df['valeur']),optimized=False).fittedvalues
    peaks = save_peaks(data_smoothed)
    # print(len(data))

    # print("Length of peaks in smoothed data:", len(peaks))

    values = df['valeur'].values
    
    peaks_unsmoothed = save_peaks(values)
    # print("Length of peaks in unsmoothed data:", len(peaks_unsmoothed))

    # Analyser les tendances et les points hauts/bas
    tend_intervals = analyze_tend_intervals( peaks,df,TDate)
    
    qualif_tend_intervals =qualification(tend_intervals)
    evenements=Evenement(tend_intervals,index)
    
    evenements_tries_par_ref = {}
    
    # Parcourir chaque événement
    for evenement in evenements:
        ref = evenement['Ref']
        # Si la référence n'existe pas encore dans le dictionnaire, l'initialiser avec une liste vide
        if ref not in evenements_tries_par_ref:
            evenements_tries_par_ref[ref] = []
        # Ajouter l'événement à la liste correspondante
        evenements_tries_par_ref[ref].append(evenement["Date"])     



    evenement_all[column]=evenements_tries_par_ref

    

    results.append({
        "column": column,
        "tendance": tend_intervals,
        "evenement":evenements,
    })
    
 
        
   
# evenement_all["externe"]=evenement_externe
matrice,array_Causes=CalculeCausa(evenement_all,columns)
# Nombre de lignes et de colonnes
# retour_quantity
n_rows = 6
n_cols = sum(1 for element in matrice[0] if element != 0) if matrice and matrice[0] else 0


row_means=[]
# Calculer la moyenne de chaque ligne
 
for row in range(n_rows):
    row_sum = sum(matrice[row][n_rows+col] for col in range(n_cols))
    print(n_cols)
    row_mean = row_sum / n_cols
    row_means.append(row_mean)

# Calculer la moyenne de toutes les moyennes des colonnes
overall_mean = max(row_means)
# overall_mean=sum(row_means) / len(row_means)

print("Moyennes de chaque lignes:", row_means)
print("Mmax moyenne:", overall_mean)
# Close the cursor and the connection
cursor.close()
conn.close()

print("end")
