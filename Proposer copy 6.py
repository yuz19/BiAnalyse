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
        
        if row is None:
            # If the row is None, execute the second query
            cursor.execute("SELECT date FROM date LIMIT 1")
            row = cursor.fetchone()
            
        if row and len(date_interval)==2:
            # Récupérer la valeur de la colonne de date
            date_value = str(row[0])
            date_interval = [datetime.strptime(str(date_str), "%Y-%m-%d") for date_str in date_interval]
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
    def getDimension(self, cursor, table_name, col):
        dim = {}
        pk_data = {}
        banned_words = ["date_id", "time_id", "id_date", "id_time"]
        cursor.execute("SELECT DATABASE()")
        schema_name =cursor.fetchone()[0]
        print(schema_name)

        # Query to retrieve primary key columns for the specified table
        cursor.execute(f"SELECT DISTINCT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = '{schema_name}' AND column_key = 'PRI'")
        primary_keys = cursor.fetchall()

        # Extract column names from the result
        primary_key_columns = [row[0] for row in primary_keys]

        for pk_column in primary_key_columns:
            pklower = pk_column.lower() 
            if pklower not in banned_words: 
                print("Primary Key:", pk_column)
                # Select primary key values with their top high measure
                cursor.execute(f"SELECT  {pk_column} from {table_name} ORDER BY {col} ASC LIMIT 30")

                rows = cursor.fetchall()
                pk_data[pk_column] = [row[0] for row in rows]

                # Query to get tables and columns where the primary key column is a foreign key
                cursor.execute(f"SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME = '{pk_column}' AND column_key = 'PRI' AND TABLE_NAME NOT IN (SELECT TABLE_NAME FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE WHERE COLUMN_NAME = '{pk_column}' AND REFERENCED_TABLE_NAME IS NOT NULL) AND TABLE_SCHEMA = '{schema_name}'")
                row = cursor.fetchall()

                if row and (row[0][0] != "time" and row[0][0] != "date"):
                    referenced_table_name = row[0][0]
                    cursor.execute(f"SELECT DISTINCT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{referenced_table_name}' AND COLUMN_NAME NOT IN (SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{referenced_table_name}' AND COLUMN_KEY = 'PRI') AND TABLE_SCHEMA = '{schema_name}'")
                    rows = cursor.fetchall()
                    columns = [row[0] for row in rows]
                    # Define the array of banned words
                    banned_words2 = ["code_postal"]
                    # Filter the columns list based on banned words
                    filtered_columns = [column for column in columns if column.lower() not in banned_words2]
                    print("Filtered Columns:", filtered_columns)
                    # Assign filtered columns to the dimension dictionary
                    dim[referenced_table_name] = filtered_columns
            
        return dim, pk_data

    def selectDimension(self,cursor,dimension_all):
        selectMesure={}
        
        for mesure,dim_keys   in dimension_all.items():
            selectTable={}
            
            print(mesure,":",dim_keys)
            
            dim_keys_toManipulate=dim_keys
            
            for table_name,cols in dim_keys[0].items():
        
                
                column_names_str = ", ".join(cols)
                where_conditions = []
                first_pk, first_values = next(iter(dim_keys_toManipulate[1].items()))
                print(first_pk,"pk",first_values)

            
        

                # Build WHERE conditions for each primary key-value pair
                where_conditions.append(f"{first_pk} IN ({', '.join(map(lambda x: f'\'{x}\'', first_values))})")

                # select column_names_str from table_name where all value pk are in values
                # Construct the SQL query
                # print("query", f"SELECT {column_names_str} FROM {table_name} WHERE {where_conditions[0]}")
                cursor.execute( f"SELECT {column_names_str} FROM {table_name} WHERE {where_conditions[0]}")

                rows = cursor.fetchall()
                print(rows)
                if (rows):
                    selectTable[table_name]=[row for row in rows]
                # remove first element
                dim_keys_toManipulate[1].pop(first_pk)
            
            selectMesure[mesure]=selectTable
        return selectMesure
            
    def start(self, columns,TDate,date_prefrence,date_interval):
        cursor = self.conn.cursor()
        results_all=[]
        results=[]
        data_all={}
        evenement_all={}
        dimension_all={}
        date_interval=self.DynamiqueFormatDate(date_interval)
        # print(date_interval)
        for index, column in enumerate(columns):
            cursor.execute(f"SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME = '{column}'")
            row = cursor.fetchall()
            # print(row)
            table_name = row[0][0]
            # print("table name", table_name)
            dimension_all[column]=self.getDimension(cursor,table_name,column)
            
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
            # print(len(data))
    
            # print("Length of peaks in smoothed data:", len(peaks))

            values = df['valeur'].values
            
            peaks_unsmoothed = self.save_peaks(values)
            # print("Length of peaks in unsmoothed data:", len(peaks_unsmoothed))

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
            
            # TRANFER DATA
            for i, item in enumerate(data_all[column]):
                item['valeur'] = data_smoothed[i]
                
        evenement_externe={
            "noel": [
            "2011-12-25",
            "2012-12-25",
            "2013-12-25",
            "2014-12-25",
            "2015-12-25",
            "2016-12-25",
            "2017-12-25",
            "2018-12-25",
            "2019-12-25",
            "2020-12-25",
            "2021-12-25",
            "2022-12-25",
            "2023-12-25",
            "2024-12-25",
            "2025-12-25"
            ],
            "jour_de_l_an": [
            "2011-01-01",
            "2012-01-01",
            "2013-01-01",
            "2014-01-01",
            "2015-01-01",
            "2016-01-01",
            "2017-01-01",
            "2018-01-01",
            "2019-01-01",
            "2020-01-01",
            "2021-01-01",
            "2022-01-01",
            "2023-01-01",
            "2024-01-01",
            "2025-01-01"
            ],
            "halloween": [
            "2011-10-31",
            "2012-10-31",
            "2013-10-31",
            "2014-10-31",
            "2015-10-31",
            "2016-10-31",
            "2017-10-31",
            "2018-10-31",
            "2019-10-31",
            "2020-10-31",
            "2021-10-31",
            "2022-10-31",
            "2023-10-31",
            "2024-10-31",
            "2025-10-31"
            ],
            "fete_nationale": [
            "2011-07-14",
            "2012-07-14",
            "2013-07-14",
            "2014-07-14",
            "2015-07-14",
            "2016-07-14",
            "2017-07-14",
            "2018-07-14",
            "2019-07-14",
            "2020-07-14",
            "2021-07-14",
            "2022-07-14",
            "2023-07-14",
            "2024-07-14",
            "2025-07-14"
            ],
            "saint_valentin": [
            "2011-02-14",
            "2012-02-14",
            "2013-02-14",
            "2014-02-14",
            "2015-02-14",
            "2016-02-14",
            "2017-02-14",
            "2018-02-14",
            "2019-02-14",
            "2020-02-14",
            "2021-02-14",
            "2022-02-14",
            "2023-02-14",
            "2024-02-14",
            "2025-02-14"
            ],
            "yennayer": [
            "2011-01-12",
            "2012-01-12",
            "2013-01-12",
            "2014-01-12",
            "2015-01-12",
            "2016-01-12",
            "2017-01-12",
            "2018-01-12",
            "2019-01-12",
            "2020-01-12",
            "2021-01-12",
            "2022-01-12",
            "2023-01-12",
            "2024-01-12",
            "2025-01-12"
            ]
        }     
        evenement_all["externe"]=evenement_externe
        matrice,array_Causes=self.CalculeCausa(evenement_all,columns)
     
        
        
        
        results_all.append(results)
        
        # print(dimension_all)
        # print(self.selectDimension(cursor,dimension_all))
        selectDim=self.selectDimension(cursor,dimension_all)
        
        results_all.append(data_all)
        # return qualif_tend_intervals,evenements,df_json
        return results_all,columns,array_Causes,selectDim
    
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
                    "Ref":self.ref_evenement(f"{peak_type} peak of {qualif} {status}",index),
                    "Date": date_fin,
                    "Optimum":min_value if interval[f"tendance {i}"]["type"] == 'diminution' else max_value
                }
                Evenement_array.append(Evenement)
            
        return Evenement_array
    
    def ref_evenement(self, Evenement, index):
        switch = {
            "Low peak of Weak increase": f"e{index}_1",
            "Low peak of Average increase": f"e{index}_2",
            "Low peak of Important increase": f"e{index}_3",
            
            "High peak of Weak decrease": f"e{index}_4",
            "High peak of Average decrease": f"e{index}_5",
            "High peak of Important decrease": f"e{index}_6"
        }
        
        return switch.get(Evenement, "Invalid event")

    
    def CausaleD(self, evenement_all, columns):
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
    
        
    def CalculeCausa(self,evenement_all,columns):
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
                        ID_e= switch.get(f"e{index}_{i}", "Invalid event").split("of")[1]+"-"+col
                        # print("dates",col,":",f"e{index}_{i}","", evenement_all[col][f"e{index}_{i}"])
                        E = event(ID_e,col, evenement_all[col][f"e{index}_{i}"], RefEvent=f"e{index}_{i}")
                        E_array.append(E)
 

        # print("Event externe")
        for index,evenetE in evenement_all['externe'].items():
            E = event(index,"externe", evenetE, RefEvent=index)
            # print(E.ID_e)  
            E_array.append(E)
            
        matrice,array_Causes=CalculeCauslite_instance.creation_matrice_influence(E_array)
        # print(matrice)
        return matrice,array_Causes