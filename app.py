
from flask import Flask, render_template, request
import numpy as np
import joblib

model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([features])
    prediction = model.predict(final_features)

    if prediction[0] == 1:
        result = "🚀 Startup Likely to Succeed"
    else:
        result = "⚠ Startup Likely to Fail"

    return render_template('result.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
