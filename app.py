from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re

app=Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # loading the dataset
    data = pd.read_csv("Language Detection.csv")
    y = data["Language"]

    

    model = pickle.load(open("model.pkl", "rb"))
    cv = pickle.load(open("transform.pkl", "rb"))

    if request.method == "POST":
        text = request.form["username"]
        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', '', text)
        text = re.sub(r'[[]]', '', text)
        text = text.lower()
        dat = [text]
        vect = cv.transform(dat).toarray()
        my_pred = model.predict(vect)
    

    return render_template("index.html", prediction_text=" The above text is in {}".format(my_pred[0]))
    
if __name__=="__main__":
    app.run(debug=True)