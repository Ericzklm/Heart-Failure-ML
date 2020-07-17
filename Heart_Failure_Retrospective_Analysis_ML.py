import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

#find the path to our CSV file within google drive and pass it as an argument to the open call.
#the path will begin with /content/drive/My Drive/ in most cases.
CHFdata = pd.read_csv(filepath_or_buffer = "CHF Data.csv", header=1, dtype=str)
CHFdata.dtypes

#convert yes and no to True and False respectively as a bool
yesNoList = ['30 day readmission', '60 day readmission', '30 day death', '60 day death', '90 day death', 'DM', 'Hypertension',
             'Coronary Artery Disease', 'Prior Stroke / TIA / Cerebral Vascular Ischemia', 'Atrial Fibrillation', 
             'Peripheral vascular disease', 'Obstructive Sleep Apnea', 'Dialysis']
#convert 1 and 0 to True and False respectively as a bool
intBoolList = ['Urine Tox negative (0) or per history', 'Urine Tox Pos Stimulant (1)', 'Urine Tox Pos Benzo (2)',
               'Urine Tox Positive Opiate (3)', 'Urine Tox Positive THC (4)', 'Smoking Currently (Yes/ No)', 'Former Smoker',
               'Marijuana (THC)', 'Alcohol (low/high)']
#convert strings to appropirate types
convert_dict = {'Age': float, 'BMI': float, 'Echocardiogram LVEF (%)': float, 'Troponin (highest)': float, 'Hemoglobin A1C': float, 'Creat (Chem 7 within 24 hours of admission)': float, 'BNP (Initial, B-type naturetic peptide)': float, 'GFR': float, 'DM': bool, 'Prior Stroke / TIA / Cerebral Vascular Ischemia': bool, 'Atrial Fibrillation': bool, 'Peripheral vascular disease': bool, 'Obstructive Sleep Apnea': bool, 'Gender': bool} 
for i in yesNoList:
    CHFdata[i] = CHFdata[i].map({'Yes':True, 'No':False})
for j in intBoolList:
    CHFdata[j] = CHFdata[j].map({'1':True, '0':False})
CHFdata['Gender'] = CHFdata['Gender'].map({'M': False, 'F': True})
CHFdata = CHFdata.astype(convert_dict) 
CHFdata.dtypes

inputColumns = ["Age", "Gender", "BMI", "Echocardiogram LVEF (%)", "Troponin (highest)", "Hemoglobin A1C" ,"Creat (Chem 7 within 24 hours of admission)", "GFR", "BNP (Initial, B-type naturetic peptide)", "DM",  "Coronary Artery Disease", "Prior Stroke / TIA / Cerebral Vascular Ischemia", "Atrial Fibrillation", "Peripheral vascular disease", "Obstructive Sleep Apnea"]
outputColumn = '30 day readmission'

inputData = CHFdata[inputColumns]
outputData = CHFdata[outputColumn]

print(inputData)
print(outputData)

inputTrain, inputTest, outputTrain, outputTest = train_test_split(inputData, outputData, test_size = .3, random_state = 2)
print(inputData.shape)
print(inputTrain.shape)
print(outputTrain.shape)
print(inputTest.shape)
print(outputTest.shape)

model = xgb.XGBClassifier()
model.fit(inputTrain, outputTrain)

from sklearn.metrics import accuracy_score
outputPred = model.predict(inputTest)
predictions = [round(value) for value in outputPred]
# evaluate predictions
accuracy = accuracy_score(outputTest, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

trainMatrix = xgb.DMatrix(inputTrain, label=outputTrain, feature_names=inputColumns[:15])
testMatrix = xgb.DMatrix(inputTest, label=outputTest, feature_names=inputColumns[:15])
params = {'max_depth':5, 'eta':0.004, 'subsample':1.0, 'min_child_weight':1.0, 'reg_lambda':0.0, 'reg_alpha':0.0, 'objective':'binary:logistic', 'eval_metric': 'error'}
model = xgb.train(params, trainMatrix, 1000, evals=[(testMatrix, "Test")], early_stopping_rounds=200)

#2D array of parameters we will test by
param_grid = {'eta':[0.1,0.05,0.01,0.005,0.001,0.0005,0.0001], 'max_depth':np.arange(1,10,1).tolist(), 'subsample':np.arange(1,0.1,-0.1).tolist(), 'colsample_bytree':np.arange(1,0.1,-0.1).tolist(), 'min_child_weight':np.arange(1,100,10).tolist()}

#Save the best results
bestParams = {}
lowestError = 2048

for max_depth in param_grid['max_depth']:
    for eta in param_grid['eta']:
        for subsample in param_grid['subsample']:
            for colsample_bytree in param_grid['colsample_bytree']:
                for min_child_weight in param_grid['min_child_weight']:
                    cvResults = xgb.cv({'max_depth':max_depth, 'eta':eta, 'subsample':subsample, 'colsample_bytree':colsample_bytree, 'min_child_weight':min_child_weight, 'objective':'binary:logistic', 'eval_metric': 'error'}, trainMatrix, num_boost_round=600, seed=2, nfold=5, early_stopping_rounds=125)
                    if cvResults['test-{}-mean'.format('error')].min() < lowestError:
                        lowestError = cvResults['test-{}-mean'.format('error')].min()
                        bestParams = {'max_depth':max_depth, 'eta':eta, 'subsample':subsample, 'colsample_bytree':colsample_bytree, 'min_child_weight':min_child_weight, 'objective':'binary:logistic', 'eval_metric': 'error'}
                    print(lowestError)
print(bestParams)
print(lowestError)
model = xgb.train(bestParams, trainMatrix, 1000, evals=[(testMatrix, "Test")], early_stopping_rounds=200)
outputTrainPredict = model.predict(trainMatrix)
outputTestPredict = model.predict(testMatrix)

from sklearn.metrics import accuracy_score, classification_report

print("Training Accuracy: " + str(accuracy_score(outputTrain, outputTrainPredict.round())))
print("Testing Accuracy: " + str(accuracy_score(outputTest, outputTestPredict.round())) + "\n")

print(classification_report(outputTest, outputTestPredict.round()))
print("\nConfusion Matrix: ")
print(pd.crosstab(outputTest, outputTestPredict.round()))
print(xgb.plot_importance(model))
model.save_model('7-16-20v2.model')
