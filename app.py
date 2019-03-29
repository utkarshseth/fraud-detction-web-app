
from flask import Flask,render_template,url_for,request,session,redirect
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np 

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/',methods=['POST'])
def test():
	filename = request.form.get('filename')
	algorithm = request.form.get('algorithm')
	print("filename fetched is-> "+str(filename))
	print("algorithm fetched is-> " + str(algorithm))
	if algorithm=="RandomForest":
		return redirect(url_for('predict',parameters=[filename,algorithm]))
	if algorithm=="AdaBoost":
		return redirect(url_for('predict1',parameters=[filename,algorithm]))
	if algorithm=="XGB-Classifier":
		return redirect(url_for('predict1',parameters=[filename,algorithm]))



@app.route('/predict/<parameters>',methods=["GET","POST"])
def predict(parameters):
	print("inside predict not checked parameters")
	if parameters:
		print("inside predict checked parameters")
		filename=parameters[2:5]
		algorithm=parameters[7:-1]
		print("inside predict checked parameters"+str(filename))
		# --ISOLATION FOREST ML CODE
		cc1 = pd.read_csv(str(filename)+".csv")
		#cc1 = pd.read_csv("cc4.csv")
			# test the model to find ""res"&&"cr"&&"acc"
		cc = cc1.iloc[:, 1:30].columns
		ccdata = cc1[cc]
		model = open("random-forest.pkl", "rb")
		cr = open("rf-classification-report.pkl", "rb")
		a = open("rf-accuracy.pkl", "rb")
		ml_model = joblib.load(model)
		classification_report = joblib.load(cr)
		accuracy = joblib.load(a)
		model_prediction = ml_model.predict(ccdata)
		model_prediction = round(float(model_prediction))
		print("check result->  "+str(model_prediction))
		cr_array = classification_report.split(' ')
		return render_template("index.html",algo=str(algorithm),res=str(model_prediction), cr=cr_array, acc=accuracy)
	return render_template("index.html")

@app.route('/predict1/<parameters>',methods=["GET","POST"])
def predict1(parameters):
	print("inside predict not checked parameters")
	if parameters:
		print("inside predict checked parameters")
		filename=parameters[2:5]
		algorithm=parameters[7:-1]
		print("inside predict checked parameters"+str(filename))
		# --ISOLATION FOREST ML CODE
		cc1 = pd.read_csv(str(filename)+".csv")
		#cc1 = pd.read_csv("cc4.csv")
			# test the model to find ""res"&&"cr"&&"acc"
		cc = cc1.iloc[:, 1:30].columns
		ccdata = cc1[cc]
		model = open("adaboost.pkl", "rb")
		cr = open("classification-report2.pkl", "rb")
		a = open("accuracy2.pkl", "rb")
		ml_model = joblib.load(model)
		classification_report = joblib.load(cr)
		accuracy = joblib.load(a)
		model_prediction = ml_model.predict(ccdata)
		model_prediction = round(float(model_prediction))
		print("check result->  "+str(model_prediction))
		cr_array = classification_report.split(' ')
		return render_template("index.html",algo=str(algorithm),res=str(model_prediction), cr=cr_array, acc=accuracy)
	return render_template("index.html")



@app.route('/predict2/<parameters>',methods=["GET","POST"])
def predict2(parameters):
	print("inside predict not checked parameters")
	if parameters:
		print("inside predict checked parameters")
		filename=parameters[2:5]
		algorithm=parameters[7:-1]
		print("inside predict checked parameters"+str(filename))
		# --ISOLATION FOREST ML CODE
		cc1 = pd.read_csv(str(filename)+".csv")
		#cc1 = pd.read_csv("cc4.csv")
			# test the model to find ""res"&&"cr"&&"acc"
		cc = cc1.iloc[:, 1:30].columns
		ccdata = cc1[cc]
		model = open("XGB-classifier.pkl", "rb")
		cr = open("creport.pkl", "rb")
		a = open("accuracy3.pkl", "rb")
		ml_model = joblib.load(model)
		classification_report = joblib.load(cr)
		accuracy = joblib.load(a)
		model_prediction = ml_model.predict(ccdata)
		model_prediction = round(float(model_prediction))
		print("check result->  "+str(model_prediction))
		cr_array = classification_report.split(' ')
		return render_template("index.html",algo=str(algorithm),res=str(model_prediction), cr=cr_array, acc=accuracy)
	return render_template("index.html")


if __name__ == '__main__':
	app.run(debug=True)
app.secret_key="any random string"