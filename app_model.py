from flask import Flask, jsonify, request, render_template
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['DEBUG'] = True

path_base = "/home/AlbaMRM/PythonAnywhere_TC"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/retrain')
def retrain_page():
    return render_template('retrain.html')

@app.route('/api/v1/predict', methods=['GET'])
def predict():
    try:
        model_path = os.path.join(path_base, 'ad_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        bill_length_mm = request.args.get('bill_length_mm', None)
        bill_depth_mm = request.args.get('bill_depth_mm', None)
        flipper_length_mm = request.args.get('flipper_length_mm', None)
        body_mass_g = request.args.get('body_mass_g', None)
        sex = request.args.get('sex', None)
        island = request.args.get('island', None)

        if (bill_length_mm is None or bill_depth_mm is None or flipper_length_mm is None or 
                body_mass_g is None or sex is None or island is None):
            return "Args empty, the data are not enough to predict", 400

        bill_length_mm = float(bill_length_mm)
        bill_depth_mm = float(bill_depth_mm)
        flipper_length_mm = float(flipper_length_mm)
        body_mass_g = float(body_mass_g)
        sex = int(sex)
        island = int(island)

        input_data = pd.DataFrame([[bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, island]], 
                                  columns=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'island'])
        prediction = model.predict(input_data)
        
        return jsonify({'predictions': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    if os.path.exists(path_base + "/data/penguins.csv"):
        data = pd.read_csv(path_base + '/data/penguins.csv')
        X = data.drop(columns='species')
        y = data['species']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = pickle.load(open(path_base + '/ad_model.pkl','rb'))
        model.fit(X_train, y_train)
        pickle.dump(model, open(path_base + '/ad_model.pkl','wb'))

        return "Model retrained."
    else:
        return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

if __name__ == '__main__':
    app.run()