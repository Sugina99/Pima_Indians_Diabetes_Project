from flask import Flask,request,url_for,redirect,render_template
import joblib
import pandas as pd

app=Flask(__name__)

model=joblib.load("model.pkl")
scale=joblib.load("scale.pkl")
 

@app.route("/")
def LandingPage():
    return render_template("index.html")

# @app.route("/predict",methods=['POST','GET'])
# def predict():
#      Pregnancies = request.form['1']
#      Glucose = request.form['2']
#      BloodPressure = request.form['3']
#      SkinThickness = request.form['4']
#      Insulin = request.form['5']
#      BMI = request.form['6']
#      DPF = request.form['7']
#      Age = request.form['8']
#      RowDf=pd.DataFrame([pd.Series([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age])])
#      RowDf_new=pd.DataFrame(scale.transform(RowDf))
#      print(RowDf_new)

#      return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)