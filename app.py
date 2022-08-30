from flask import Flask,request,url_for,redirect,render_template
import joblib
import pandas as pd

app=Flask(__name__)

model=joblib.load("model.pkl")
scale=joblib.load("scale.pkl")
 

@app.route("/")
def hello_world():
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)