import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime
import math
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error 

class Proposer:
    def __init__(self, conn):
        self.conn = conn

    def start(self, column,TDate):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME = '{column}'")
        row = cursor.fetchall()
        print(row)
        table_name = row[0][0]
        print("table name", table_name)
        data_frames=[]
        # Récupérer les données de la colonne spécifiée
        # cursor.execute(f"SELECT {column} FROM {table_name}")
        cursor.execute(f"SELECT jour, mois, annee, SUM({column}) FROM {table_name},time WHERE time.date_ID={table_name}.date_ID GROUP BY annee, mois, jour ORDER BY annee, mois, jour ASC")
        rows = cursor.fetchall()
        
        data_frames = [row[3] for row in rows]
        
        data = []

        for row in rows:
            datasave = {
                'index_JMA': f"{row[0]}-{row[1]}-{row[2]}",
                'index_MA': f"{row[1]}-{row[2]}",
                'index_A': f"{row[2]}",
                'valeur': row[3]
            }
            data.append(datasave)


        # Créer un DataFrame à partir des données
        # df = pd.DataFrame(data_frames,columns=[column])
        # df[column] = df[column].astype(float)
        # # print(data)
        df = pd.DataFrame(data)
        # df['valeur'] = df['valeur'].astype(float)
        df['valeur'] = pd.to_numeric(df['valeur'], errors='coerce')
        print(df)
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
        evenements=self.Evenement(tend_intervals)
        
        # delta_t  Différence en jours
        delta_t=self.delta_t(evenements)
        # print(delta_t)
        
        ecart_temporel=self.ecart_temporel(evenements, delta_t ,0.9)
        print(ecart_temporel)
        # # # Convertir le DataFrame en un format JSON compatible
        # Convertir le DataFrame en une liste de dictionnaires
        df_json = df.to_json(orient='split')
        # print(tend_intervals)
        return qualif_tend_intervals,evenements,df_json
    
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

    def Evenement(self, tend_intervals):
        
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
                "Ref":self.ref_evenement(f"{peak_type} peak of {qualif} {status}"),
                "Date": date_fin,
                "Optimum":min_value if interval[f"tendance {i}"]["type"] == 'diminution' else max_value
            }
            Evenement_array.append(Evenement)
            
        return Evenement_array
    
    def ref_evenement(self, Evenement):
        switch = {
            "Low peak of Weak increase": 'e1',
            "High peak of Weak increase": 'e2',
            "Low peak of Average increase": 'e3',
            "High peak of Average increase": 'e4',
            "Low peak of Important increase": 'e5',
            "High peak of Important increase": 'e6',
            
            "Low peak of Weak decrease": 'e7',
            "High peak of Weak decrease": 'e8',
            "Low peak of Average decrease": 'e9',
            "High peak of Average decrease": 'e10',
            "Low peak of Important decrease": 'e11',
            "High peak of Important decrease": 'e12'
        }
        
        return switch.get(Evenement, "Invalid event")
    
    def ecart_temporel(self,evenements, delta_t, E):
        
        ecart_temporel = {}
        for i in range(1, 13):
            en = f"e{i}"
            Occe = self.occurance(evenements, en)
            print(en,":",Occe)
            if Occe!=0:
                ecart = self.calculate_N(Occe, delta_t, E)
                ecart_temporel[en] = ecart
            else:
                ecart_temporel[en]="null"
                
        return ecart_temporel
    
    def calculate_N(self,Occe, delta_t, E):
        # Calcul de t0
        t0 = sum(delta_t) / (Occe - 1)
        print("TO",t0)
        # Calcul de ψ0
        psi_0 = 1 - E
        
        # Calcul de η
        eta = -math.log(1 - E) / t0
        print("eta",eta)
        # Calcul de la probabilité
        proba = math.exp(-eta * t0)
        print(-eta * t0)
        print("proba",proba)
        print("\n")
        return proba
    
    def delta_t(self,evenements):
        
        # Convertir les chaînes de dates en objets de date
        dates = [datetime.strptime(evenement["Date"], "%d-%m-%Y") for evenement in evenements]

        # Calculer les écarts temporels entre les événements consécutifs
        ecarts_temporels = []
        for i in range(len(dates) - 1):
            ecart_temporel = (dates[i + 1] - dates[i]).days  # Différence en jours
            ecarts_temporels.append(ecart_temporel)

        return ecarts_temporels     
    def occurance(self,evenements,en):
        count=0
        for evenement in evenements:
            if evenement["Ref"]==en:
                count+=1
          
        return count  
       