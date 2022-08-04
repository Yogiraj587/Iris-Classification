import numpy as np
from flask import  Flask, request, jsonify, render_template
import pickle

app  = Flask(__name__)
model  = pickle.load(open('ML/IRIS/iris.pkl','rb'))

@app.route("/")

def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data1 = request.form['Sepal length']
    data2 = request.form['Sepal width']
    data3 = request.form['Petal length']
    data4 = request.form['Petal width']

    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('predicted.html',pred=pred)

if __name__ =="__main__":
    app.run(debug=True)
    
