import csv
import math
import multiprocessing 
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from functools import partial

def getParams(inputData, outputData, inputColumns, maxDepth): #, trainMatrix):
    inputTrain, inputTest, outputTrain, outputTest = train_test_split(inputData, outputData, test_size = .3, random_state = 2)
    trainMatrix = xgb.DMatrix(inputTrain, label=outputTrain, feature_names=inputColumns[:15])
    bestParams = {}
    lowestError = 2048
    for eta in [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]:
        for subsample in np.arange(1,0.1,-0.1).tolist():
            for colsample_bytree in np.arange(1,0.1,-0.1).tolist():
                for min_child_weight in np.arange(1,100,10).tolist():
                    cvResults = xgb.cv({'max_depth':maxDepth, 'eta':eta, 'subsample':subsample, 'colsample_bytree':colsample_bytree, 'min_child_weight':min_child_weight, 'objective':'binary:logistic', 'eval_metric': 'error'}, trainMatrix, num_boost_round=600, seed=2, nfold=5, early_stopping_rounds=125)
                    if cvResults['test-{}-mean'.format('error')].min() < lowestError:
                        lowestError = cvResults['test-{}-mean'.format('error')].min()
                        bestParams = {'max_depth':maxDepth, 'eta':eta, 'subsample':subsample, 'colsample_bytree':colsample_bytree, 'min_child_weight':min_child_weight, 'objective':'binary:logistic', 'eval_metric': 'error'}
                    print(lowestError)
    return [lowestError, bestParams]

if __name__ == '__main__':
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

    inputTrain, inputTest, outputTrain, outputTest = train_test_split(inputData, outputData, test_size = .3, random_state = 2)

    model = xgb.XGBClassifier()
    model.fit(inputTrain, outputTrain)

    outputPred = model.predict(inputTest)
    predictions = [round(value) for value in outputPred]
    # evaluate predictions
    accuracy = accuracy_score(outputTest, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    trainMatrix = xgb.DMatrix(inputTrain, label=outputTrain, feature_names=inputColumns[:15])
    testMatrix = xgb.DMatrix(inputTest, label=outputTest, feature_names=inputColumns[:15])
    params = {'max_depth':5, 'eta':0.004, 'subsample':1.0, 'min_child_weight':1.0, 'reg_lambda':0.0, 'reg_alpha':0.0, 'objective':'binary:logistic', 'eval_metric': 'error'}
    model = xgb.train(params, trainMatrix, 1000, evals=[(testMatrix, "Test")], early_stopping_rounds=200)

    pool = multiprocessing.Pool() 
    inputs = np.arange(1,10,2).tolist() #, trainMatrix]
    func = partial(getParams, inputData, outputData, inputColumns)
    outputs_async = pool.map_async(func, inputs) 
    resultArr = outputs_async.get() 

    #for max_depth in param_grid['max_depth']:
      #resultArr.append(getParams(param_grid,max_depth))
    #print(resultArr)

    lowestError = resultArr[0][0]
    bestParams = resultArr[0][1]

    for y in resultArr:
      if y[0] < lowestError:
        lowestError = y[0]
        bestParams = y[1]
    print(lowestError)
    print(bestParams)
      
    model = xgb.train(bestParams, trainMatrix, 1000, evals=[(testMatrix, "Test")], early_stopping_rounds=200)
    outputTrainPredict = model.predict(trainMatrix)
    outputTestPredict = model.predict(testMatrix)

    print("Training Accuracy: " + str(accuracy_score(outputTrain, outputTrainPredict.round())))
    print("Testing Accuracy: " + str(accuracy_score(outputTest, outputTestPredict.round())) + "\n")

    print(classification_report(outputTest, outputTestPredict.round()))
    print("\nConfusion Matrix: ")
    print(pd.crosstab(outputTest, outputTestPredict.round()))
    print(xgb.plot_importance(model))
    model.save_model('oopsForgotToChangeThis.model')
