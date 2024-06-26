import math
import pandas as pd
import os
from datetime import datetime,timedelta
class CalculeCauslite:
    def __init__(self):
        self.DI = 0
    def save_event_data(self,event_data, filename):
        event_df = pd.DataFrame(event_data)

        # Check if the file exists
        if os.path.exists(filename):
            # Append mode (a) to add new data without overwriting
            event_df.to_csv(filename, index=False, mode='a', header=False)
        else:
            # Write header if the file doesn't exist
            event_df.to_csv(filename, index=False, header=True)  

    def creation_matrice_influence(self, E):
        matrice = [[0 for _ in range(len(E))] for _ in range(len(E))]
        evenement_csv = {"evenement": []}

        # e1_1 e1_2 e1_3 e2_1 e2_2 e2_3
        # Remplir la matrice avec les valeurs de causalité
        for i in range(len(E)):
            for j in range(len(E)):
                if i != j and E[j].Measure!="externe":
                    matrice[i][j] = self.calculer_DI_causal_2(E[i], E[j], 0.9)

        # Affichage de la matrice de causalité
        print("--------la matrice de causalité")
        for row in matrice:
            print(" ".join(map(str, row)))
        # Affichage de la matrice de causalité avec les événements Ei et Ej
        print("--------la matrice de causalité")
        print("    ", end="")
        for i in range(len(E)):
            print(E[i].RefEvent, end=" ")
        print()  # New line
        for i in range(len(E)):
            print(E[i].RefEvent, end="")
            for j in range(len(E)):
                print(matrice[i][j], end=" ")
            print()  # New line

        # Interprétation de la matrice
        print("-------Interprétation de la matrice")
        for i in range(len(E)):
            evenement = f"{E[i].ID_e}"

            for j in range(len(E)):
                if matrice[i][j] != 0 and matrice[i][j]< 1:
                    print(E[i].ID_e + " causes " + E[j].ID_e + " with an Influence Degree of: " + str(matrice[i][j]))
                    if(matrice[i][j]>0,5 and E[i].Measure!=E[j].Measure):
                        evenement+=f",{E[j].ID_e}"
            evenement_csv["evenement"].append(evenement)

        self.save_event_data(evenement_csv,"events.csv")
        #SEND TO FRONT
        # Trier la matrice par ordre descendant
        flattened_matrice = [(i, j, matrice[i][j]) for i in range(len(matrice)) for j in range(len(matrice[i]))]
        sorted_matrice = sorted(flattened_matrice, key=lambda x: x[2], reverse=True)
        array_Causes=[]
        print("++++ creation array avec relations de causalite : des : ++++")
        for i in range(len(sorted_matrice)):
            index_i, index_j, influence_degree = sorted_matrice[i]
            array_Causes.append({
                
                    "causes":f"{E[index_i].ID_e} causes {E[index_j].ID_e} with an Influence Degree of:" ,
                    "degree":influence_degree,
                    # "mesure":E[index_i].Measure
                    "mesure": "externe" if E[index_i].Measure == "externe" or E[index_j].Measure == "externe" else E[index_i].Measure if E[index_i] .Measure!= "externe" else E[index_i] .Measure
                })
        
        # Afficher les 10 premières valeurs de causalité
        # print("--------Top 10 des relations de causalité:")
        # for i in range(min(10, len(sorted_matrice))):
        #     index_i, index_j, influence_degree = sorted_matrice[i]
        #     print(f"{E[index_i].ID_e} causes {E[index_j].ID_e} with an Influence Degree of: {influence_degree}")
        return  matrice,array_Causes

    def calculer_DI_causal_2(self, E1_cause, E2_effect,rate_E):
        i = 0
        sum = 0
        j = 0
        t = 0

        if E1_cause.Measure == E2_effect.Measure:
            self.DI = 0
        else:
            if len(E1_cause.pos_dates[0].split("-"))==3:
                E1_pos_dates = sorted(E1_cause.pos_dates, key=lambda x: datetime.strptime(x, '%Y-%m-%d'), reverse=True)

                
                E2_pos_dates = sorted(E2_effect.pos_dates, key=lambda x: datetime.strptime(x, '%Y-%m-%d'), reverse=True)
                
                E1_pos_dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in E1_pos_dates]
            
                E2_pos_dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in E2_pos_dates]
            elif len(E1_cause.pos_dates[0].split("-"))==2:
                E1_pos_dates = sorted(E1_cause.pos_dates, key=lambda x: datetime.strptime(x, '%Y-%m'), reverse=True)

                
                E2_pos_dates = sorted(E2_effect.pos_dates, key=lambda x: datetime.strptime(x, '%Y-%m'), reverse=True)
                
                E1_pos_dates = [datetime.strptime(date_str, '%Y-%m') for date_str in E1_pos_dates]
            
                E2_pos_dates = [datetime.strptime(date_str, '%Y-%m') for date_str in E2_pos_dates]
            elif  len(E1_cause.pos_dates[0].split("-"))==1:
                E1_pos_dates = sorted(E1_cause.pos_dates, key=lambda x: datetime.strptime(x, '%Y'), reverse=True)

                
                E2_pos_dates = sorted(E2_effect.pos_dates, key=lambda x: datetime.strptime(x, '%Y'), reverse=True)
                
                E1_pos_dates = [datetime.strptime(date_str, '%Y') for date_str in E1_pos_dates]
            
                E2_pos_dates = [datetime.strptime(date_str, '%Y') for date_str in E2_pos_dates]

            # print(E2_effect.RefEvent,"after",E2_pos_dates)
            end1 = len(E2_pos_dates)                
            T0 = 0

            if end1 == 1:
                # T0 = E2_pos_dates[0] ?????
                T0=1
            else:
                T0 = (E2_pos_dates[0] - E2_pos_dates[end1 - 1]) / (end1 - 1)          
                # print("T0",T0)
                
                T0=self.convertToDate(str(T0))
                # print(E2_effect.Measure,":",E2_effect.RefEvent,",",E2_pos_dates[0],'',E2_pos_dates[end1 - 1],"T0 changed",T0)
            Mu = (-math.log(1 -rate_E)) / T0
        
            for i in range(len(E2_pos_dates)):
                while t == 0 and j < len(E1_pos_dates):
                    if E2_pos_dates[i] > E1_pos_dates[j]:
                        # sum += math.exp((-Mu) * ( self.convertToDate(str( E2_pos_dates[i] - E1_pos_dates[j] )) ) )
                        ti=E2_pos_dates[i] - E1_pos_dates[j]
                        # print("time",ti)
                        ti=self.convertToDate(str(ti))
                        sum += math.exp((-Mu) * (ti))
                        t = 1
                    else:
                        j += 1
                t = 0
            self.DI = sum / end1 if end1 != 0 else 0
            # print("mu",Mu,"DI",self.DI)
            
        return self.DI
        
    def convertToDate(self, duration_str):
        # Parse the duration string into a timedelta object
        duration = timedelta(days=0, seconds=0, microseconds=0)
        parts = duration_str.split(", ")
        for part in parts:
            if "days" in part:
                duration += timedelta(days=int(part.split()[0]))
            elif "day" in part:
                duration += timedelta(days=int(part.split()[0]))
            else:
                time_parts = part.split(":")
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds = 0
                if len(time_parts) > 2:
                    seconds = int(time_parts[2].split(".")[0])
                microseconds = 0
                if len(time_parts) > 2 and "." in time_parts[2]:
                    microseconds = int(time_parts[2].split(".")[1])
                duration += timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)

        # Convert the timedelta object to days
        total_days = duration.days + duration.seconds / (24 * 3600) + duration.microseconds / (24 * 3600 * 10**6)
        # print("Total days:", total_days)

        return total_days
