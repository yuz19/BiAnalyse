import pandas as pd
import numpy as np
import mysql.connector
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error 
class Proposer:
    def __init__(self, conn):
        self.conn = conn

    def start(self, column,TDate):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME = '{column}'")
        row = cursor.fetchall()
        table_name = row[0][0]
        print("table name", table_name)

        # Récupérer les données de la colonne spécifiée
        cursor.execute(f"SELECT  SUM({column}) FROM {table_name},time WHERE time.date_ID={table_name}.date_ID GROUP BY annee, mois, jour ORDER BY annee, mois, jour ASC")

        rows = cursor.fetchall()

        # Créer un DataFrame à partir des données
        df = pd.DataFrame(rows, columns=[column])
        print(df)
        
        # Détecter les points hauts et bas
        # values = df[column].values
        # Perform smoothing on the data
        # alpha = 0.2  # Smoothing parameter
        data_smoothed = SimpleExpSmoothing(df[column].astype(float)).fit(smoothing_level=self.training_alpha(df[column].astype(float)),optimized=False).fittedvalues
        peaks = self.save_peaks(data_smoothed)
        print("Length of peaks in smoothed data:", len(peaks))

        values = df[column].values
        peaks_unsmoothed = self.save_peaks(values)
        print("Length of peaks in unsmoothed data:", len(peaks_unsmoothed))

        # Analyser les tendances et les points hauts/bas
        tend_intervals = self.analyze_tend_intervals( peaks)

        # # # Convertir le DataFrame en un format JSON compatible
        # Convertir le DataFrame en une liste de dictionnaires
        df_json = df.to_json(orient='split')
        # print(tend_intervals)
        return tend_intervals,rows
    
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

    def analyze_tend_intervals(self, peaks):
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

            interval = {
                f"tendance {index}": {
                    "type": trend_type,
                    "interval": [
                        {
                            "index": prev_peak_index,
                            "value": prev_peak_value
                        },
                        {
                            "index": curr_peak_index,
                            "value": curr_peak_value
                        }
                    ]
                }
            }

            tend_intervals.append(interval)
            index += 1  # Increment index for the next trend

        return tend_intervals
