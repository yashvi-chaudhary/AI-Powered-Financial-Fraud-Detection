from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)


# Load Model and Scaler

model = joblib.load(
    "models/fraud_detection_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get values from form

        features = []

        for i in range(30):
            value = float(request.form[f"feature{i}"])
            features.append(value)


        # Convert to numpy array

        data = np.array(features).reshape(1,-1)


        # Scale input

        data = scaler.transform(data)


        # Prediction

        prediction = model.predict(data)


        if prediction[0] == 1:
            result = "⚠️ Fraudulent Transaction Detected"

        else:
            result = "✅ Legitimate Transaction"


        return render_template(
            "index.html",
            prediction=result
        )


    except Exception as e:

        return str(e)



if __name__ == "__main__":
    app.run(debug=True)