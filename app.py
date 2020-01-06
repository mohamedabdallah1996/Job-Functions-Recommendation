# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 22:13:15 2019

@author: USER
"""

from flask import Flask, jsonify
from utils import normalize
import pickle
import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to Flask App'

@app.route('/<title>', methods=['GET'])
def recommend_job_finctions(title):
    # recommend job functions
    model = pickle.load(open('pickle/model.pkl', 'rb'))
    vectorizer = pickle.load(open('pickle/vectorizer.pkl', 'rb'))
    classes = pickle.load(open('pickle/classes.pkl', 'rb'))
    
    title_norm = normalize(title)
    title_vec = vectorizer.transform([title_norm])
    pred = model.predict(title_vec)
  
    # return job functions from the prediction
    results = []
    for index in np.where(pred[0]==1)[0]:
        results.append(classes[index])

    if(len(results) > 0):
        return jsonify(results)
    else:
        return 'Sorry, System are not able to recommend job function for this title !!'


if(__name__ == '__main__'):
    try:
        app.run(debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin!!")
    