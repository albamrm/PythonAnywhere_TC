from flask import Flask, jsonify, request, render_template
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import subprocess

app = Flask(__name__)
app.config['DEBUG'] = True

path_base = '/home/AlbaMRM/PythonAnywhere_TC'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/retrain')
def retrain_page():
    return render_template('retrain.html')

# Cargar el modelo
def load_model():
    model_path = os.path.join(path_base, 'ad_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar el escalador
def load_scaler():
    scaler_path = os.path.join(path_base, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# Cargar el mapeo de categorías
def load_mappings():
    mappings_path = os.path.join(path_base, 'mappings.pkl')
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    return mappings

# Enruta la funcion al endpoint /api/v1/predict
@app.route('/api/v1/predict', methods = ['GET'])
def predict():
    try:
        model = load_model()
        scaler = load_scaler()
        mappings = load_mappings()

        # Obtener los parámetros de la solicitud GET
        bill_length_mm = request.args.get('bill_length_mm', None)
        bill_depth_mm = request.args.get('bill_depth_mm', None)
        flipper_length_mm = request.args.get('flipper_length_mm', None)
        body_mass_g = request.args.get('body_mass_g', None)
        sex = request.args.get('sex', None)
        island = request.args.get('island', None)

        # Verificar que todos los parámetros estén presentes
        if (bill_length_mm is None or bill_depth_mm is None or flipper_length_mm is None or 
                body_mass_g is None or sex is None or island is None):
            return jsonify({'error': 'Args empty, the data are not enough to predict'}), 400

        # Convertir los parámetros a sus tipos adecuados
        try:
            bill_length_mm = float(bill_length_mm)
            bill_depth_mm = float(bill_depth_mm)
            flipper_length_mm = float(flipper_length_mm)
            body_mass_g = float(body_mass_g)
            sex = int(sex)
            island = int(island)
        except ValueError:
            return jsonify({'error': 'Invalid input types'}), 400

        # Crear un DataFrame con los datos de entrada
        input_data = pd.DataFrame([[bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, island]], 
                                  columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'island'])

        # Asegurarse de que las columnas están en el mismo orden que las usadas durante el entrenamiento
        expected_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'island']
        input_data = input_data[expected_columns]

        # Escalar los datos de entrada
        input_data_scaled = load_scaler().transform(input_data)

        # Realizar la predicción
        prediction = model.predict(input_data_scaled)

        # Mapear la predicción al nombre de la especie
        species = load_mappings()['species'][prediction[0]]

        # Retornar la predicción en formato JSON
        return jsonify({'predictions': species})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint para reentrenar el modelo
@app.route('/api/v1/retrain', methods = ['GET'])
def retrain():
    if os.path.exists(os.path.join(path_base, 'data', 'penguins.csv')):
        data = pd.read_csv(os.path.join(path_base, 'data', 'penguins.csv'))
        
        # Separar características y variable objetivo
        X = data.drop(columns = 'species')
        y = data['species']
        
        # Escalar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)
        model = KNeighborsClassifier(n_neighbors=5)
        
        # Entrenar el modelo
        model.fit(X_train, y_train)
        
        # Guardar el modelo reentrenado
        with open(os.path.join(path_base, 'ad_model_new.pkl'), 'wb') as f:
            pickle.dump(model, f)
        
        # Guardar el escalador
        with open(os.path.join(path_base, 'scaler_new.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        return 'Model retrained successfully.'
    else:
        return '<h2>New data for retrain NOT FOUND. Nothing done!</h2>'
    
@app.route('/webhook_2024', methods = ['POST'])
def webhook():
    # Ruta al repositorio donde se realizará el pull
    path_repo = '/home/AlbaMRM/sabadosteam'
    servidor_web = '/var/www/albamrm_pythonanywhere_com_wsgi.py'

    # Comprueba si la solicitud POST contiene datos JSON
    if request.is_json:
        payload = request.json
        # Verifica si la carga útil (payload) contiene información sobre el repositorio
        if 'repository' in payload:
            # Extrae el nombre del repositorio y la URL de clonación
            repo_name = payload['repository']['name']
            clone_url = payload['repository']['clone_url']
            
            # Cambia al directorio del repositorio
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify({'message': 'El directorio del repositorio no existe'}), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(['git', 'pull'], check = True)
                subprocess.run(['touch', servidor_web], check = True)
                return jsonify({'message': f'Se realizó un git pull en el repositorio {repo_name}'}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error al realizar git pull en el repositorio {repo_name}'}), 500
        else:
            return jsonify({'message': 'No se encontró información sobre el repositorio en la carga útil (payload)'}), 400
    else:
        return jsonify({'message': 'La solicitud no contiene datos JSON'}), 400

if __name__ == '__main__':
    app.run()