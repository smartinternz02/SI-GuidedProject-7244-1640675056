import requests
import pandas as pd
import numpy as np
import pickle
import os
# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "eDL3wevXqN3-F217noMwED-UXfhymaNRSBJWF1CXz8Be"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
from flask import Flask,request,render_template
import numpy as np
import  pickle

#create flask app
app=Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,5)
    loaded_model = pickle.load(open("HIC1.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route("/predict",methods=["POST"])
def predict():
    predict_list=[float(x) for x in request.form.values()]
    result=ValuePredictor(predict_list)
# NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"fields": [['age','sex','bmi','children','smoker','region']], "values": [[23,'male',22.2,2,'yes','southeast']]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/insurance/predictions?version=2022-02-01', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
    predictions =response_scoring.json()
    print(predictions)
    pred = predictions['predictions'][0]['values'][0][0]

    #prediction = loaded_model.predict(features_value)
    #output=prediction[0]
    #print(output)
    print(pred)
    return render_template("home.html",prediction_text="Premium cost is Rs %f"%result)
if __name__ == '__main__':
      app.run(debug=True)