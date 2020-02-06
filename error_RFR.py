#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
import csv
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#Load the Dataset 
df = pd.read_csv("C:/Users/aldosamn/Desktop/Project/Final_Descriptors_Pro.csv", index_col=0)
target = df['Entropy']
features = df[df.columns[3:]]

# Define search space
n_estimators=[50,100,200,300,400,500,600,700,800,900,1000]
param_grid = dict(n_estimators=n_estimators)
# Define outer folds
kFolds = KFold(n_splits=10, shuffle=True, random_state=9).split(X=features.values, y=target.values)
# Define inner folds
GS = GridSearchCV(RandomForestRegressor( max_features='auto', n_jobs=-1,random_state=9), param_grid,
                  cv=KFold(n_splits=10, shuffle=True, random_state=9),
                           n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')

# Open results file and write out headers
out_file = open("E:/Project/My Scripts/Results/GridSearch_RFR.csv", 'w')
wr = csv.writer(out_file, dialect='excel')
headers = ["n_estimators", "r2", "error_ma", "error_ms", "error_rms", "error_mp", "error_max"]
wr.writerow(headers)
out_file.flush()

for index_train, index_test in kFolds:
    # Get train and test splits
    x_train, x_test = features.iloc[index_train].values, features.iloc[index_test].values
    y_train, y_test = target.iloc[index_train].values, target.iloc[index_test].values

    # Apply min max normalization
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Fit
    GS.fit(x_train, y_train)
    
     # Get best params
    best_params = GS.best_params_

    # Calculate error metrics
    predictions = GS.predict(x_test)
    diff = y_test - predictions
    r2 = r2_score(y_test, predictions)
    error_ma = mean_absolute_error(y_test, predictions)
    error_ms = mean_squared_error(y_test, predictions)
    error_rms = np.sqrt(np.mean(np.square(diff)))
    error_mp = np.mean(abs(np.divide(diff, y_test))) * 100
    error_max = np.amax(np.absolute(diff))

    # Write results
    row = [best_params['n_estimators'], r2,
           error_ma, error_ms, error_rms, error_mp, error_max]
    wr.writerow(row)
    out_file.flush()

out_file.close()


# In[ ]:




