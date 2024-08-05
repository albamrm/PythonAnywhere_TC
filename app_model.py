from flask import Flask, jsonify, request, render_template
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['DEBUG'] = True

path_base = os.getenv('PATH_BASE', '/home/AlbaMRM/PythonAnywhere_TC')
model_path = os.path.join(path_base, 'ad_model.pkl')

# Cargar el modelo una vez al iniciar la aplicación
model = None
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/retrain')
def retrain_page():
    return render_template('retrain.html')

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        required_fields = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'island']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing fields in request data'}), 400
        
        # Convert input data to float/int as required
        try:
            input_data = {
                'bill_length_mm': float(data['bill_length_mm']),
                'bill_depth_mm': float(data['bill_depth_mm']),
                'flipper_length_mm': float(data['flipper_length_mm']),
                'body_mass_g': float(data['body_mass_g']),
                'sex': int(data['sex']),
                'island': int(data['island'])
            }
        except ValueError as e:
            return jsonify({'error': f'Invalid data type: {e}'}), 400
        
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        
        return jsonify({'predictions': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/retrain', methods=['POST'])
def retrain():
    try:
        data_path = os.path.join(path_base, 'data/penguins.csv')
        
        if not os.path.exists(data_path):
            return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>", 404
        
        data = pd.read_csv(data_path)
        X = data.drop(columns='species')
        y = data['species']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train)
        
        new_model_path = os.path.join(path_base, 'ad_model_new.pkl')
        with open(new_model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return "Model retrained successfully."
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()