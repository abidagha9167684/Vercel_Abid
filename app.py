import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)

model2 = pickle.load(open("Vercel_Check\Depdata.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("DASS.html")



@flask_app.route("/Depression", methods = ["POST"])
def Depression():
    return render_template("Depression.html")



@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model2.predict(features)
    return render_template("Depression.html", prediction_text = format(prediction))



if __name__ == "__main__":
    flask_app.run(debug=True)
