import csv
import math
import multiprocessing 
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from functools import partial

#function finds the best set of parameters which gives the lowest amount of error
def getParams(inputData, outputData, inputColumns, maxDepth):
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
    #read in the data from file
    print(multiprocessing.cpu_count())
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
    #dictionary of type conversions
    convert_dict = {'Age': float, 'BMI': float, 'Echocardiogram LVEF (%)': float, 'Troponin (highest)': float, 'Hemoglobin A1C': float, 'Creat (Chem 7 within 24 hours of admission)': float, 'BNP (Initial, B-type naturetic peptide)': float, 'GFR': float, 'DM': bool, 'Prior Stroke / TIA / Cerebral Vascular Ischemia': bool, 'Atrial Fibrillation': bool, 'Peripheral vascular disease': bool, 'Obstructive Sleep Apnea': bool, 'Gender': bool} 

    for i in yesNoList:
        CHFdata[i] = CHFdata[i].map({'Yes':True, 'No':False})
    for j in intBoolList:
        CHFdata[j] = CHFdata[j].map({'1':True, '0':False})
    #Convert Gender to a bool
    CHFdata['Gender'] = CHFdata['Gender'].map({'M': False, 'F': True})
    CHFdata = CHFdata.astype(convert_dict) 
    #CHFdata.dtypes

    #specify which factors/columns we want to consider in training and what column represents the outcome
    inputColumns = ["Age", "Gender", "BMI", "Echocardiogram LVEF (%)", "Troponin (highest)", "Hemoglobin A1C" ,"Creat (Chem 7 within 24 hours of admission)", "GFR", "BNP (Initial, B-type naturetic peptide)", "DM",  "Coronary Artery Disease", "Prior Stroke / TIA / Cerebral Vascular Ischemia", "Atrial Fibrillation", "Peripheral vascular disease", "Obstructive Sleep Apnea"]
    outputColumn = '30 day readmission'

    inputData = CHFdata[inputColumns]
    outputData = CHFdata[outputColumn]

    #split the data into a set for training and a set for testing
    inputTrain, inputTest, outputTrain, outputTest = train_test_split(inputData, outputData, test_size = .3, random_state = 2)

    #test the data by running a quick training session and verify reasonable accuracy
    model = xgb.XGBClassifier()
    model.fit(inputTrain, outputTrain)

    outputPred = model.predict(inputTest)
    predictions = [round(value) for value in outputPred]
    accuracy = accuracy_score(outputTest, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    #a slightly longer and more accurate training session
    trainMatrix = xgb.DMatrix(inputTrain, label=outputTrain, feature_names=inputColumns[:15])
    testMatrix = xgb.DMatrix(inputTest, label=outputTest, feature_names=inputColumns[:15])
    params = {'max_depth':5, 'eta':0.004, 'subsample':1.0, 'min_child_weight':1.0, 'reg_lambda':0.0, 'reg_alpha':0.0, 'objective':'binary:logistic', 'eval_metric': 'error'}
    model = xgb.train(params, trainMatrix, 1000, evals=[(testMatrix, "Test")], early_stopping_rounds=200)

    to get the best results, hyper parameter selection is employed which will run a session for each combination of parameter values by calling the getParams function. This section takes a significant amount of time to run so parallelism is used.
    pool = multiprocessing.Pool(processes = 4) 
    inputs = np.arange(1,10,1).tolist()
    func = partial(getParams, inputData, outputData, inputColumns)
    outputs_async = pool.map_async(func, inputs) 
    resultArr = outputs_async.get()

    #resultArr =[]
    #for z in inputs:
        #resultArr.append(getParams(inputData, outputData, inputColumns, z))

    lowestError = resultArr[0][0]
    bestParams = resultArr[0][1]

    #get the parameters that resulted in the least amount of error
    for y in resultArr:
      if y[0] < lowestError:
        lowestError = y[0]
        bestParams = y[1]
    print(lowestError)
    print(bestParams)

    #train using the best parameters to form the model
    model = xgb.train(bestParams, trainMatrix, 1000, evals=[(testMatrix, "Test")], early_stopping_rounds=200)
    outputTrainPredict = model.predict(trainMatrix)
    outputTestPredict = model.predict(testMatrix)

    #output the results for the model
    print("Training Accuracy: " + str(accuracy_score(outputTrain, outputTrainPredict.round())))
    print("Testing Accuracy: " + str(accuracy_score(outputTest, outputTestPredict.round())) + "\n")

    print(classification_report(outputTest, outputTestPredict.round()))
    print("\nConfusion Matrix: ")
    print(pd.crosstab(outputTest, outputTestPredict.round()))
    print(xgb.plot_importance(model))

    #save the model for later use
    model.save_model('7-17-20.model')
