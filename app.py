#import relevant libraries for flask, html rendering and loading the ML model
from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scale = joblib.load("scale.pkl")


@app.route("/")
def landingPage():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    
    pregnancies = request.form['1']
    glucose = request.form['2']
    bloodPressure = request.form['3']
    skinThickness = request.form['4']
    insulin = request.form['5']
    bmi = request.form['6']
    dpf = request.form['7']
    age = request.form['8']

    rowDF= pd.DataFrame([pd.Series([pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,dpf,age])])
    rowDF_new = pd.DataFrame(scale.transform(rowDF))

    print(rowDF_new)

    # model prediction
    prediction=model.predict_proba(rowDF_new)
    print(f" The predicted value is : {prediction[0][1]}")
    
    if prediction[0][1]>=0.5:
        val_pred=round(prediction[0][1],3)
        print(f"the round value is {val_pred*100}%")
        return render_template('result.html',pred=f"You have a chance of having diabetes.\n\n Probability of being a diabetic is {val_pred*100}%.\n\n Advice: Exercise Regularly")
    else:
        val_pred=round(prediction[0][0],3)
        print(f"the round value is {val_pred*100}%")
        return render_template('result.html',pred=f"Congratulations!!!, You are safe zone.\n\n Probability of being a non-diabetic is {val_pred*100}%.")

if __name__ == '__main__':
    app.run(debug=True)
    