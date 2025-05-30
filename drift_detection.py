import pandas as pd
import json
from sklearn import datasets

from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('data.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



reference_data = pd.read_csv('reference_data.csv')
new_data = pd.read_csv('new_data.csv')

import numpy as np


def introduce_drift(data, drift_features, drift_amount=0.1, random_seed=42):
    np.random.seed(random_seed)
    drifted_data = data.copy()
    
    for feature in drift_features:
        if feature in data.columns:
            drifted_data[feature] += np.random.normal(loc=0, scale=drift_amount, size=data.shape[0])
    
    return drifted_data
    
features_to_drift = ['Glucose', 'BloodPressure', 'SkinThickness', 'Pregnancies']

drifted_data = introduce_drift(X_test, features_to_drift, drift_amount=50)

reference_data['Outcome'] = y_train.reset_index(drop = True)
drifted_data['Outcome'] = y_test.reset_index(drop = True)

drifted_data.to_csv('new_data.csv', index=False)
reference_data.to_csv('reference_data.csv', index=False)

## Combine data drift and the reference data

pd.concat([reference_data, drifted_data]).reset_index(drop =True).to_csv('combined_data.csv', index=False)
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(current_data=drifted_data, reference_data=reference_data, column_mapping=None)



import datetime

today_date = datetime.datetime.today().strftime("%Y-%m-%d")
data_drift_report.save_html('reports\data_drift_report_{today_date}.html')

from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(current_data=drifted_data.drop('Outcome', axis =1), reference_data=reference_data.drop('Outcome', axis =1), column_mapping=None)
report_json = data_drift_report.as_dict()
drift_detected = report_json['metrics'][0]['result']['dataset_drift']

print(drift_detected)
with open("drift_flag.json", "w") as f:
    json.dump({"drift_detected": True}, f)
