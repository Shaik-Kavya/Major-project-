#import necessary packages 
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import warnings 
warnings.filterwarnings('ignore') 

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb')) 

@app.route('/')
def home():
    return render_template('home.html') 

@app.route('/index')
def index():
    return render_template('home.html') 

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html') 

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features=[]
    X=request.form['x-axis']
    int_features.append(float(X)) 
    Y=request.form['y-axis'] 
    int_features.append(float(Y)) 
    month=request.form['month'] 
    int_features.append(float(month)) 
    day=request.form['day'] 
    int_features.append(float(day)) 
    ffmc=request.form['ffmc'] 
    int_features.append(float(ffmc)) 
    dmc=request.form['dmc'] 
    int_features.append(float(dmc)) 
    dc=request.form['dc'] 
    int_features.append(float(dc)) 
    isi=request.form['isi']
    int_features.append(float(isi)) 
    temperature=request.form['temperature'] 
    int_features.append(float(temperature)) 
    rh=request.form['rh'] 
    int_features.append(float(rh)) 
    wind=request.form['wind']
    int_features.append(float(wind)) 
    rain=request.form['rain'] 
    int_features.append(float(rain)) 
    
    final_features=[np.array(int_features)] 

    prediction = model.predict(final_features)
    
    output = float(prediction[0]) 
    if(output>0):
        result = 'OCCURRENCE OF FIRE'
    else:
        result = 'NO OCCURRENCE OF FIRE'
    
    return render_template('prediction.html', prediction_result=result)

if __name__ =="__main__":
    app.run(debug=True)
