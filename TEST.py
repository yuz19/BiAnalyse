import pandas as pd
import plotly.graph_objects as go
import pymysql
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error

# Se connecter à la base de données MySQL
connection = pymysql.connect(host='localhost', user='root', password='1962', database='ventes_enligne')

# Extraire les données de la base de données
query = 'SELECT prix_ventes FROM ventes'
data = pd.read_sql(query, connection)

# Fermer la connexion à la base de données
connection.close()

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

# Entraînement du modèle final avec la meilleure valeur de alpha
final_model = SimpleExpSmoothing(data).fit(smoothing_level=best_alpha)
smoothed_data = final_model.fittedvalues

# Générer le graphe pour les données originales
fig_original = go.Figure()
fig_original.add_trace(go.Scatter(x=data.index, y=data['prix_ventes'], mode='lines', name='Prix de ventes (Original)'))
initial_range = [data.index.min(), data.index.min() + 500]  # Ajustez selon vos besoins
fig_original.update_layout(title='Série Prix de ventes au fil du temps (Original)', xaxis_title='Index', yaxis_title='Prix de ventes', xaxis=dict(range=initial_range))

# Générer le graphe pour les données lissées
fig_smoothed = go.Figure()
fig_smoothed.add_trace(go.Scatter(x=data.index, y=smoothed_data, mode='lines', name='Prix de ventes (Lissé)'))
fig_smoothed.update_layout(title='Série Prix de ventes au fil du temps (Lissé)', xaxis_title='Index', yaxis_title='Prix de ventes', xaxis=dict(range=initial_range))

# Afficher les graphiques
fig_original.show()
fig_smoothed.show()