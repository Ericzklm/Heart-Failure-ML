Heart Failure Machine Learning

The goal of this project is to use machine learning tools in order to perform retrospective analysis on
patients admitted to the hospital for congestive heart failure (CHF) and develop a model to accurately
predict hospital readmittance and/or mortality. The data itself is not published here but the code used 
is present. This project is an expansion of the following: 
https://github.com/Michael-Equi/hfr/blob/master/notebooks/hfr-ml-comparison.ipynb.

For this application, we use XGBoost to perform gradient boosting. XGBoost in particular was used
due to the relative efficiency and accuracy on smaller datasets like ours. Additionally, XGBoost
tools can efficiently deal with missing entries that occur within our dataset. The main way
we train the model is with cross validation to find the optimal parameters via hyperparameterization
that will result in the lowest amount of error between the training data and testing data.
Other tools are used to store data and perform data analysis

This study's intended usage is in medical papers and hospital practice.

Data Entry Fields:
	ID, admission date, age, gender, BMI, zip code, echocardiogram LVEF %,
	Troponin (highest), Hemoglobin A1C, Creat (Chem 7 within 24 hours of admission),
	GFR, BNP (Initial, B-type naturetic peptide), Urine Tox negative (0) or per history,
	Urine Tox Pos Benzo (2), Urine Tox Positive Opiate (3), Urine Tox Positive THC (4),
	Smoking Currently (Yes/ No), Former Smoker, Smoking (Pack Year History), 
	Marijuana (THC), Alcohol (low/high), 30 day readmission, 60 day readmission,
	30 day death, 60 day death, 90 day death, DM, Hypertension, Coronary Artery Disease,
	Prior Stroke / TIA / Cerebral Vascular Ischemia, Atrial Fibrillation, 
	Peripheral vascular disease, Obstructive Sleep Apnea, Aortic Stenosis, Dialysis












