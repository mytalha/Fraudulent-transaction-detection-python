from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the trained model
model = LogisticRegression()
model = joblib.load("model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the form inputs
    trans_amt = float(request.form["trans_amt"])
    hour = int(request.form["hour"])
    history_30 = float(request.form["history_30"])
    city_pop = float(request.form["city_pop"])
    distance = float(request.form["distance"])

    # Create the input data point
    data_point = pd.DataFrame({
        'amt': [trans_amt],
        'hour': [hour],
        'history_30': [history_30],
        'interaction_30': [history_30 / trans_amt],
        'city_pop': [city_pop],
        'distance': [distance]
    })

    # Make the prediction
    prediction = model.predict(data_point)[0]

    # Determine the output message
    if prediction == 1:
        output_message = "The transaction is classified as fraudulent."
        prediction_class = "fraud"

    else:
        output_message = "The transaction is classified as legitimate."
        prediction_class = "legitimate"


    return render_template("result.html", prediction=output_message)

if __name__ == "__main__":
    app.run()
