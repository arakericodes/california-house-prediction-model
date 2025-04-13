import pandas as pd
import numpy as np
from  sklearn.preprocessing import PolynomialFeatures
from flask import Flask, request, jsonify, render_template
import pickle
application = Flask(__name__)
app = application

Linear_Model = pickle.load(open('regressor.pkl', 'rb'))
Scaler_Model = pickle.load(open('scaler.pkl', 'rb'))


@app.route("/")
def greet():
    return render_template('index.html')

@app.route("/Predict-Prices", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        MedInc = float(request.form.get('mi'))
        HouseAge = float(request.form.get('hage'))
        AveRooms = float(request.form.get('avr'))	
        AveBedrms	= float(request.form.get('avb'))
        Population	= float(request.form.get('pop'))
        AveOccup	= float(request.form.get('avo'))
        Latitude	= float(request.form.get('lat'))
        Longitude= float(request.form.get('long'))
        new_data = Scaler_Model.transform([[MedInc,	HouseAge	,AveRooms	,AveBedrms,	Population	,AveOccup	,Latitude	,Longitude]])
        result = Linear_Model.predict(new_data)
        return render_template('home.html', result=result)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0')