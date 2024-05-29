import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = os.getenv('MODEL_PATH', 'model.joblib')
model = joblib.load(model_path)

@app.route('/score', methods=['POST'])
def score():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if file:
        data = pd.read_csv(file)
        predictions = model.predict(data)
        data['y_pred'] = predictions
        return jsonify(data.to_dict(orient='records'))
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
