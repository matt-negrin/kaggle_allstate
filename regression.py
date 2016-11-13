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
        
# selecting columns to run regression
total_rows = len(train.columns.format())
starting_spot = np.where(train.columns=='cont1')[0][0]
model_train = train.ix[:,starting_spot:total_rows]

# building a training and validation set
indexes = np.random.rand(len(train)) < 0.8
model_train = train[indexes]
model_valid = train[~indexes]

# preping training and validation for regression
from sklearn import linear_model
x_train = model_train.drop('loss', axis=1)
x_valid = model_valid.drop('loss', axis=1)

y_train = model_train['loss']
y_valid = model_valid['loss']

print('x_train: ', len(x_train))

# Create linear regression object
# regr = linear_model.LinearRegression()

# Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)
