import json
import mysql.connector
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from Granger import Granger
from Proposer import Proposer
from TableFait import TableFait




from flask import Flask, request, jsonify, Response
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/api/analyse/*": {"origins": "*"}})
CORS(app, resources={r"/api/sql/*": {"origins": "*"}})
CORS(app, resources={r"/api/getMesure/*": {"origins": "*"}})


array_return=[]
@app.route('/api/analyse/', methods=['GET', 'POST'])
def analyse():
    global array_return

    if request.method == 'POST':
        array_return=[]

        data = request.json
        columns = data.get('columns', [])
        algorithms = data.get('algorithms', [])
        
        date_prefrence=data.get('date_prefrence')
        
        if date_prefrence=="jour": 
            date_prefrence="annee, mois, jour"
        elif date_prefrence=="mois":
            date_prefrence="annee, mois"
        else:
            date_prefrence="annee"
        # :["12/01/2011","30/01/2013"],or NULL
        date_interval=data.get('date_interval')
        
        print(date_prefrence)
        print(date_interval)
           
        if any(algorithms):
            if algorithms.get('granger', False):
                
                Granger_instance=Granger(columns,conn) 
                granger_result = Granger_instance.start(date_prefrence,date_interval)
                if granger_result:
                    array_return.append({'granger': granger_result})
                else:
                    array_return.append({'granger':[]})
            else:    
                array_return.append({'granger':[]})       
                    
                       
            if algorithms.get('proposer', False):
                # column = data.get('column', [])
                column="prix_ventes"
                TDate="index_JMA"
                proposer_instance=Proposer(conn)
                proposer_result_tend_intervals,proposer_evenement,proposer_result_df = proposer_instance.start(column,TDate,date_prefrence,date_interval)
                
                if proposer_result_tend_intervals :
                    array_return.append({'proposer': [{"tendance": proposer_result_tend_intervals,"evenement":proposer_evenement, "value": proposer_result_df}]})
                else:
                    array_return.append({'proposer':[]})
                    
            if array_return:
                return jsonify(array_return)
            else:
                return jsonify({"message": "No results found"}), 404
        
        
        else:
            return jsonify({"Error": "Choisir un algorithme"}), 400
    
    elif request.method == 'GET':
        return jsonify(array_return), 200
    else:
        return jsonify({"Error": "Choisir un algorithme"}), 400

 
@app.route('/api/getMesure/', methods=['GET'])
def getMesure():
    mesure=[]
    TableFait_instance=TableFait(conn)
    mesure.append(TableFait_instance.get_mesure())
    return mesure

# Connexion MySQL
conn = None
connection_info={}

@app.route('/api/sql/', methods=['GET', 'POST'])
def connect_to_mysql():
    global conn
    global connection_info
    if request.method == 'POST':
      
        data = request.json
        hostname = data.get('host')
        dbname = data.get('database')
        root = data.get('user')
        password = data.get('password')
        port = data.get('port')

        try:
            conn = mysql.connector.connect(
                host=hostname,
                database=dbname,
                user=root,
                password=password,
                port=port
            )
            connection_info = {
                'host': hostname,
                'database': dbname,
                'user': root,
                'password': password,
                'port': port
            }
            if conn.is_connected():
                return jsonify({'message': 'Connexion réussie à MySQL'}), 200
            else:
                return jsonify({'error': 'Impossible de se connecter à MySQL'}), 500
        except mysql.connector.Error as e:
            return jsonify({'error': f'Erreur de connexion à MySQL : {str(e)}'}), 500
        # else:
        #     print("test")
        #     try:
        #         conn = mysql.connector.connect(
        #             host=connection_info['host'],
        #             database=connection_info['database'],
        #             user=connection_info['user'],
        #             password=connection_info['password'],
        #             port=connection_info['port']
        #         )
        #         if conn.is_connected():
        #             return jsonify({'message': 'Connexion réussie à MySQL'}), 200
        #         else:
        #             return jsonify({'error': 'Impossible de se connecter à MySQL'}), 500
        #     except mysql.connector.Error as e:
        #         return jsonify({'error': f'Erreur de connexion à MySQL : {str(e)}'}), 500
         
    elif request.method == 'GET':
        if connection_info:
            return jsonify(connection_info), 200
        else:
            return jsonify({'error': 'Aucune connexion MySQL établie'}), 404
    else:
        return jsonify({'error': 'Méthode non autorisée'}), 405


# Reconnexion MySQL
@app.route('/api/resql/', methods=['POST'])
def reconnect_to_mysql():
    global conn
    global connection_info
    if request.method == 'POST':
        data = request.json
        hostname = data.get('host')
        dbname = data.get('database')
        root = data.get('user')
        password = data.get('password')
        port = data.get('port')
         
        try:
            if conn and conn.is_connected():
                conn.close()
             
            conn = mysql.connector.connect(
                host=hostname,
                database=dbname,
                user=root,
                password=password,
                port=port
            )
            if conn.is_connected():
                
                connection_info = {
                    'host': hostname,
                    'database': dbname,
                    'user': root,
                    'password': password,
                    'port': port
                }
                return jsonify({'message': 'Reconnexion réussie à MySQL avec de nouvelles informations de connexion'})
            else:
                return jsonify({'error': 'Impossible de se reconnecter à MySQL avec de nouvelles informations de connexion'})
        except mysql.connector.Error as e:
            return jsonify({'error': f'Erreur de reconnexion à MySQL avec de nouvelles informations de connexion : {str(e)}','check':'sqlError'})

    else:
        return jsonify({'error': 'Méthode non autorisée'}), 405

if __name__ == '__main__':
    app.run(port=8000,debug=True)
