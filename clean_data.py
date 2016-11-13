## loading relevant packages
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing

def load_dataframe_from_csv(frame_name='data/train.csv'):
    return pd.read_csv(frame_name)

def cleaned_dataframes_from_file(train_name='data/train.csv', seed=1):
    np.random.seed(seed)
    # master dataset
    master_train = load_dataframe_from_csv(train_name)
    run_from_dataframe(master_train)
    
def cleaned_dataframes_from_dataframe(master_train, seed=1):
    np.random.seed(seed)
    
    # copies of master
    train = pd.DataFrame.copy(master_train)
    encoded_frame = encode_categorical_data(train)
    loss = train['loss']
    
    return split_data(encoded_frame)
    
def select_columns(dataframe, col_type):
    headers = dataframe[:0].columns.values.T.tolist()
    col_headers = [header for header in headers if col_type in header]
    column_indicies = [i for i in range(len(headers)) if col_type in headers[i]]
    return (col_headers, dataframe[column_indicies])
    
def encode_categorical_data(dataframe):
    cont_headers, cont_frame = select_columns(dataframe, 'cont')
    cat_headers, cat_frame = select_columns(dataframe, 'cat')
    result_frame = pd.concat([dataframe.ix[:, 0], dataframe['loss'], cont_frame], axis=1)

    # feature engineering
    for i in range(len(cat_headers)):
        encoded_dataframe = onehot_encoder(cat_frame, cat_headers[i])
        result_frame = pd.concat([result_frame, encoded_dataframe], axis=1)
        
    return result_frame

def split_data(dataframe):
    # building a training and validation set
    indexes = np.random.rand(len(dataframe)) < 0.8
    return (dataframe[indexes], dataframe[~indexes])

def onehot_encoder(dataframe, column):
    encoder = preprocessing.LabelEncoder()
    return pd.get_dummies(dataframe[column]).rename(columns=lambda x: column + '_' + str(x))