        # # Calculer l'écart type pour chaque deux points consécutifs
        # df['Omega'] = df[column].diff().abs().rolling(window=2).std()
        

        # # Identifier l'ordre idéal (n) qui minimise Omega
        # min_omega = df['Omega'].min()
        # ideal_order = df[df['Omega'] == min_omega].index.values[0] + 1  # Ajouter 1 pour convertir l'indice en ordre

        # print("Ordre idéal (n) :", ideal_order)
        # print("Omega minimal :", min_omega)
        
        # # Récupérer les valeurs x et y pour ajuster le polynôme
        # x = df.index.values
        # y = df[column].values

        # # Ajuster un polynôme de degré idéal_order aux données
        # coefficients = np.polyfit(x, y, ideal_order)
        
        # # Utiliser les coefficients pour créer les tendances
        # trend_values = np.polyval(coefficients, x)
        # # print("TREND",trend_values)