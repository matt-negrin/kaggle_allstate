## loading relevant packages
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import linear_model
from clean_data import load_dataframe_from_csv
from clean_data import cleaned_dataframes_from_dataframe

def fit_from_file(file_name='data/train.csv'):
    master_train, master_validation = cleaned_dataframes_from_file()
    return fit(master_train, master_validation)
    
def fit(master_train, master_validation):
    # Creating copies of initial train and valid datasets
    output_train = pd.DataFrame.copy(master_train)
    output_validation = pd.DataFrame.copy(master_validation)

    #copy split data
    mut_train = pd.DataFrame.copy(master_train)
    mut_validation = pd.DataFrame.copy(master_validation)

    # Split the data into training/testing sets
    x_train = mut_train.drop('loss', axis=1)
    x_validation = mut_validation.drop('loss', axis=1)

    #remove id cols
    mut_train.drop('id', axis=1)
    mut_validation.drop('id', axis=1)

    # Split the targets into training/testing sets
    y_train = master_train['loss']
    y_validation = master_validation['loss']

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Adding predicted loss and error columns to initial datasets
    output_train['predicted_loss'] = regr.predict(x_train)
    output_validation['predicted_loss'] = regr.predict(x_validation)
    output_train['error'] = output_train['predicted_loss'] - output_train['loss']
    output_validation['error'] = output_validation['predicted_loss'] - output_validation['loss']
    
    return (output_train, output_validation, regr)



