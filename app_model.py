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
        island = request.args.get('island', None)
        bill_length_mm = request.args.get('bill_length_mm', None)
        bill_depth_mm = request.args.get('bill_depth_mm', None)
        flipper_length_mm = request.args.get('flipper_length_mm', None)
        body_mass_g = request.args.get('body_mass_g', None)
        sex = request.args.get('sex', None)
        
        # Verificar que todos los parámetros estén presentes
        if (island is None or bill_length_mm is None or bill_depth_mm is None or flipper_length_mm is None or 
                body_mass_g is None or sex is None):
            return jsonify({'error': 'Args empty, the data are not enough to predict'}), 400

        # Convertir los parámetros a sus tipos adecuados
        try:
            island = int(island)
            bill_length_mm = float(bill_length_mm)
            bill_depth_mm = float(bill_depth_mm)
            flipper_length_mm = float(flipper_length_mm)
            body_mass_g = float(body_mass_g)
            sex = int(sex)            
        except ValueError:
            return jsonify({'error': 'Invalid input types'}), 400

        # Crear un DataFrame con los datos de entrada
        input_data = pd.DataFrame([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]], 
                                  columns = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'])

        # Asegurarse de que las columnas están en el mismo orden que las usadas durante el entrenamiento
        expected_columns = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
        input_data = input_data[expected_columns]

        # Escalar los datos de entrada
        input_data_scaled = scaler.transform(input_data)

        # Realizar la predicción
        prediction = model.predict(input_data_scaled)

        # Mapear la predicción al nombre de la especie
        species = mappings['species'][prediction[0]]

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
        model = KNeighborsClassifier(n_neighbors = 5)
        
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
        return 'New data for retrain NOT FOUND. Nothing done!'

if __name__ == '__main__':
    app.run()