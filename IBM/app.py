import numpy as np
import pickle
import pandas
import os
import joblib
import requests
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
API_KEY = "P74qHZDx-fCbTIrheslp3qDIxCpQ2QUOtoMMYIyTJrqK"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
@app.route('/')# route to display the home page
def home():
    return render_template('index.html') #rendering the home page

@app.route('/predict',methods=["POST","GET"])# route to show the predictions in a web UI
def predict():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values() ]  
    features_values=[np.array(input_feature)]
    names = [['Id','Sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation','Settlement size']]
    data = pandas.DataFrame(features_values,columns=names)
    #data = scale.fit_transform(features_values)
    payload_scoring = {"input_data": [{"fields": ['Id','Sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation','Settlement size'], "values": features_values}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/dd880536-c906-44c9-a0af-e1d0b18f9197/predictions?version=2023-02-25', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    prediction=response_scoring.json()
     # predictions using the loaded model file
    if (prediction == 0):
       return render_template("index.html",prediction_text ="Not a potential customer")
    elif (prediction == 1):
       return render_template("index.html",prediction_text = "Potential customer")
    else:
       return render_template("index.html",prediction_text = "Highly potential customer")
     # showing the prediction results in a UI
if __name__=="__main__":
    
    # app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)