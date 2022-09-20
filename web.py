from pyexpat.errors import messages
from tkinter import N
from flask import Flask, render_template, request, url_for, redirect, session
import pandas as pd
import numpy as np
from joblib import load

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = '7110c8ae51a4b5af97be6534caef90e4bb9bdcb3380af008f90b23a5d1616bf319bc298105da20fe'

normal = pd.read_csv('data/normal.csv')
encoded = pd.read_csv('data/encoded.csv')
labels = normal.columns

categorical = []
for l in labels:
    if normal[l].dtypes == object:
        normal[l] = normal[l].astype('string')
        categorical.append(l)

rf = load('data/car-predict.joblib')

@app.route('/', methods=['GET','POST'])
def index():
    message = '$-'
    errors = {}
    if request.method == 'POST':
        data = {k:request.form.get(k) for k in labels}

        val = validation(data)

        if len(val)>0:
            errors = val
        else:
            message = '${}'.format(round(do_predict(data),3))

    messages = {'price':message}
    session['messages'] = messages
    return render_template('index.html', data=[{'transmission':'Mecánica'},{'transmission':'Automática'},{'transmission':'Automática secuencial'}], messages=messages, errors=errors)

def validation(data):
    errors = {}
    for k, v in data.items():
        each = []
        if v == '':
            errors[k] = 'Debe llenar el campo.'
            continue
        if k in categorical and len(normal.loc[normal[k].str.lower() == str(v).lower()])<1:
            errors[k] = 'No conocemos esto.'

    return errors

def do_predict(data):
    X = {}
    for k, v in data.items():
        if k in categorical:
            i = normal.loc[normal[k].str.lower() == str(v).lower()].index[0]
            X[k] = encoded[k].iloc[i]
        else:
            X[k] = v

    df = pd.DataFrame(X, index=[0])

    return rf.predict(df)[0]

if __name__ == '__main__':
    app.run(debug=True)