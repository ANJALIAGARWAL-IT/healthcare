from flask import Flask,render_template, url_for ,flash , redirect
from sklearn.externals import joblib
from flask import request
import numpy as np
import tensorflow
import os
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import pickle
import sys
import glob
import re

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

app=Flask(__name__,template_folder='template')


pickle_in = open("finalized_model.pkl","rb")
classifier=pickle.load(pickle_in)

dir_path = os.path.dirname(os.path.realpath(__file__))


def model_predict(full_path,model):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255 
    prediction = model.predict(data)
    return prediction
#FOR THE SECOND MODEL
@app.route("/")

@app.route("/home")
def home():
    return render_template("home.html")
 


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/covid")
def covid():
    return render_template("covid.html")
@app.route("/cancer")
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
def diabetes():
    
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/liver")
def liver():
    
    return render_template("liver.html")

@app.route("/kidney")
def kidney():
    
    return render_template("kidney.html")



def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8):#Diabetes
        loaded_model = joblib.load("model1")
        result = loaded_model.predict(to_predict)
    elif(size==30):#Cancer
        loaded_model = joblib.load("model")
        result = loaded_model.predict(to_predict)
    elif(size==12):#Kidney
        loaded_model = joblib.load("model3")
        result = loaded_model.predict(to_predict)
    elif(size==10):
        loaded_model = joblib.load("model4")
        result = loaded_model.predict(to_predict)
    elif(size==11):#Heart
        loaded_model = joblib.load("model2")
        result =loaded_model.predict(to_predict)
    elif(size==5): 
        result=classifier.predict(to_predict)
    
    return result[0]

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list)==30):#Cancer
            result = ValuePredictor(to_predict_list,30)
        elif(len(to_predict_list)==8):#Daiabtes
            result = ValuePredictor(to_predict_list,8)
        elif(len(to_predict_list)==12):
            result = ValuePredictor(to_predict_list,12)
        elif(len(to_predict_list)==11):
            result = ValuePredictor(to_predict_list,11)

        elif(len(to_predict_list)==10):
            result = ValuePredictor(to_predict_list,10)
        elif(len(to_predict_list)==5):
            result=ValuePredictor(to_predict_list,5)

            
    if(int(result)==1):
        prediction='Sorry ! Suffering'
    else:
        prediction='Congrats ! you are Healthy' 
    return(render_template("result.html", prediction=prediction))


if __name__ == "__main__":
    app.run(debug=True)
