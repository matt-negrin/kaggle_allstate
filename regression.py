## loading relevant packages
import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
import random

# master dataset
master_train = pd.read_csv('data/train.csv')
master_test = pd.read_csv('data/test.csv')

# copies of master
train = pd.DataFrame.copy(master_train)
test  = pd.DataFrame.copy(master_test)

# feature engineering
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

def onehot_encoder(dataframe, column):
    new_frame = pd.get_dummies(dataframe[column]).rename(columns=lambda x: column + '_' + str(x))
    return pd.concat([dataframe, new_frame], axis=1)

for column in train.columns:
    if 'cat' in column:
        train = onehot_encoder(train, column)

# building a training and validation set
indexes = np.random.rand(len(train)) < 0.8
idx_model_train = train[indexes]
idx_model_valid = train[~indexes]
        
# selecting columns to run regression
def column_selector(dataframe):
    total_rows = len(dataframe.columns.format())
    starting_spot = np.where(dataframe.columns=='cont1')[0][0]
    output = dataframe.ix[:,starting_spot:total_rows]
    return(output)

model_train = column_selector(idx_model_train)
model_valid = column_selector(idx_model_valid)

# preping training and validation for regression
from sklearn import linear_model

# Split the data into training/testing sets
x_train = model_train.drop('loss', axis=1)
x_valid = model_valid.drop('loss', axis=1)

# Split the targets into training/testing sets
y_train = model_train['loss']
y_valid = model_valid['loss']

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Creating copies of initial train and valid datasets
output_train = pd.DataFrame.copy(idx_model_train)
output_valid = pd.DataFrame.copy(idx_model_valid)

# Adding predicted loss and error columns to initial datasets
output_train['predicted_loss'] = regr.predict(x_train)
output_valid['predicted_loss'] = regr.predict(x_valid)
output_train['error'] = output_train['predicted_loss'] - output_train['loss']
output_valid['error'] = output_valid['predicted_loss'] - output_valid['loss']