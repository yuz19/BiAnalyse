  
    def ecart_temporel(self,evenements, delta_t, E):
        
        ecart_temporel = {}
        for i in range(1, 13):
            en = f"e{i}"
            Occe = self.occurance(evenements, en)
            # print(en,":",Occe)
            if Occe!=0:
                ecart = self.calculate_N(Occe, delta_t, E)
                ecart_temporel[en] = ecart
            else:
                ecart_temporel[en]="null"
                
        return ecart_temporel
    
    def calculate_N(self,Occe, delta_t, E):
        # Calcul de t0
        t0 = sum(delta_t) / (Occe - 1)
        print("TO",t0,"\n")
        # Calcul de ψ0
        psi_0 = 1 - E
        
        # Calcul de η
        eta = -math.log(1 - E) / t0
        
        # Calcul de la probabilité
        proba = math.exp(-eta * t0)
        
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
       