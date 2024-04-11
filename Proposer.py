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
        
class Proposer:
    def __init__(self, conn):
        self.conn = conn
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
    def start(self, columns,TDate,date_prefrence,date_interval):
        cursor = self.conn.cursor()
        results_all=[]
        results=[]
        data_all={}
        evenement_all={}
        date_interval=self.DynamiqueFormatDate(date_interval)
        # print(date_interval)
        for index, column in enumerate(columns):
            cursor.execute(f"SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME = '{column}'")
            row = cursor.fetchall()
            # print(row)
            table_name = row[0][0]
            # print("table name", table_name)
            data_frames=[]

            
            # Récupérer les données de la colonne spécifiée
            # cursor.execute(f"SELECT {column} FROM {table_name}")
            # cursor.execute(f"SELECT jour, mois, annee, SUM({column}) FROM {table_name},time WHERE time.date_ID={table_name}.date_ID GROUP BY annee, mois, jour ORDER BY annee, mois, jour ASC")
            if len(date_interval)==0:
                cursor.execute(f"SELECT {date_prefrence}, SUM({column}) FROM {table_name},time WHERE time.date_ID={table_name}.date_ID GROUP BY {date_prefrence} ORDER BY {date_prefrence} ASC")
            else:
                # print('havebet')
                cursor.execute(f"SELECT {date_prefrence}, SUM({column}) FROM {table_name},time WHERE time.date_ID={table_name}.date_ID and time.date BETWEEN  '{date_interval[0]}' AND '{date_interval[1]}' GROUP BY {date_prefrence} ORDER BY {date_prefrence} ASC")

            rows = cursor.fetchall()
            
            # data_frames = [row[3] for row in rows]
            
            data = []

            for row in rows:
            
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
                # datasave = {
                #     'index_JMA': f"{row[0]}-{row[1]}-{row[2]}",
                #     'index_MA': f"{row[1]}-{row[2]}",
                #     'index_A': f"{row[2]}",
                #     'valeur': row[3]
                # }
                data.append(datasave)
               
     
                if column in data_all:
                        data_all[column].append(datasave)
                else:
                    data_all[column] = [datasave]
 
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
            data_smoothed = SimpleExpSmoothing(df['valeur']).fit(smoothing_level=self.training_alpha(df['valeur']),optimized=False).fittedvalues
            peaks = self.save_peaks(data_smoothed)
            print(data_smoothed)
            print("Length of peaks in smoothed data:", len(peaks))

            values = df['valeur'].values
            
            peaks_unsmoothed = self.save_peaks(values)
            print("Length of peaks in unsmoothed data:", len(peaks_unsmoothed))

            # Analyser les tendances et les points hauts/bas
            tend_intervals = self.analyze_tend_intervals( peaks,df,TDate)
            
            qualif_tend_intervals =self.qualification(tend_intervals)
            evenements=self.Evenement(tend_intervals,index)
            
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
            
            
        matrice,array_Causes=self.CalculeCausa(evenement_all,columns)
     
        
        
        
        results_all.append(results)
        
        results_all.append(data_all)
        # return qualif_tend_intervals,evenements,df_json
        return results_all,columns,array_Causes
    
    def training_alpha(self,data):
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

    def save_peaks(self, values):
        peaks = {}
        peaks[0] = {"index": 0, "value": values[0]}
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                peaks[i] = {"index": i, "value": values[i]}
            elif values[i] < values[i - 1] and values[i] < values[i + 1]:
                peaks[i] = {"index": i, "value": values[i]}
        return peaks
    # Tendance + amplitude +date_debut +date_fin
    def analyze_tend_intervals(self, peaks,df,TDate):
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
    
    def qualification(self, tend_intervals):
     
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

    def Evenement(self, tend_intervals,index):
        
        Evenement_array = []
        for i, interval in enumerate(tend_intervals, 1):
            qualif = interval[f"tendance {i}"]["qualification"]
            peak_type =  "Low" if interval[f"tendance {i}"]["type"] == 'diminution' else "High"
            status = "decrease" if interval[f"tendance {i}"]["type"] == 'diminution' else "increase"
            date_fin = interval[f"tendance {i}"]['interval'][1]['date-fin']  # Accéder à la clé 'date-fin' du deuxième élément de la liste 'interval'
            max_value = max([abs(value["value"]) for i, value in enumerate(interval[f"tendance {i}"]['interval'], 1)])
            min_value = min([abs(value["value"]) for i, value in enumerate(interval[f"tendance {i}"]['interval'], 1)])
            

            
            Evenement = {
                "Evenement": f"{peak_type} peak of {qualif} {status}",
                "Ref":self.ref_evenement(f"{peak_type} peak of {qualif} {status}",index),
                "Date": date_fin,
                "Optimum":min_value if interval[f"tendance {i}"]["type"] == 'diminution' else max_value
            }
            Evenement_array.append(Evenement)
            
        return Evenement_array
    
    def ref_evenement(self, Evenement, index):
        switch = {
            "Low peak of Weak increase": f"e{index}_1",
            "High peak of Weak increase": f"e{index}_2",
            "Low peak of Average increase": f"e{index}_3",
            "High peak of Average increase": f"e{index}_4",
            "Low peak of Important increase": f"e{index}_5",
            "High peak of Important increase": f"e{index}_6",
            
            "Low peak of Weak decrease": f"e{index}_7",
            "High peak of Weak decrease": f"e{index}_8",
            "Low peak of Average decrease": f"e{index}_9",
            "High peak of Average decrease": f"e{index}_10",
            "Low peak of Important decrease": f"e{index}_11",
            "High peak of Important decrease": f"e{index}_12"
        }
        
        return switch.get(Evenement, "Invalid event")

    
    def CausaleD(self, evenement_all, columns):
        DI_allEvenet = []
        for idx1, col1 in enumerate(columns):
            print (idx1)
            print(col1)
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
                                    print(col1, ',', i, '-', col2, ',', j, ":", di_test)
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
                                    print(col2, ',', i, '-', col1, ',', j, ":", di_test)
                                    event_DI2.append((f"e{idx2}_{i}-e{idx1}_{j}", di_test))

                    DI_allEvenet.append({
                        "columns": [col1, col2],
                        "evenet_DI sens1": event_DI1,
                        "evenet_DI sens2": event_DI2
                    })

        return DI_allEvenet
    
        
    def CalculeCausa(self,evenement_all,columns):
        E_array = []


        CalculeCauslite_instance=CalculeCauslite()
        for index, col in enumerate(columns): 
            switch = {
                f"e{index}_1": "Low peak of Weak increase",
                f"e{index}_2": "High peak of Weak increase",
                f"e{index}_3": "Low peak of Average increase",
                f"e{index}_4": "High peak of Average increase",
                f"e{index}_5": "Low peak of Important increase",
                f"e{index}_6": "High peak of Important increase",

                f"e{index}_7": "Low peak of Weak decrease",
                f"e{index}_8": "High peak of Weak decrease",
                f"e{index}_9": "Low peak of Average decrease",
                f"e{index}_10": "High peak of Average decrease",
                f"e{index}_11": "Low peak of Important decrease",
                f"e{index}_12": "High peak of Important decrease"
            }
         
            for i in range(1, 13):        
                if f"e{index}_{i}" in evenement_all[col]:
                        ID_e= switch.get(f"e{index}_{i}", "Invalid event").split("of")[1]+"-"+col
                        print("dates",col,":",f"e{index}_{i}","", evenement_all[col][f"e{index}_{i}"])
                        E = event(ID_e,col, evenement_all[col][f"e{index}_{i}"], RefEvent=f"e{index}_{i}")
                        E_array.append(E)
      
        matrice,array_Causes=CalculeCauslite_instance.creation_matrice_influence(E_array)
        print(matrice)
        return matrice,array_Causes