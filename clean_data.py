## loading relevant packages
import pandas as pd
import numpy as np
import sklearn as sk
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

def load_dataframe_from_csv(frame_name='data/train.csv'):
    return pd.read_csv(frame_name)

def cleaned_dataframes_from_file(train_name='data/train.csv', seed=1, encoder = ce.OneHotEncoder()):
    np.random.seed(seed)
    # master dataset
    master_train = load_dataframe_from_csv(train_name)
    cleaned_dataframes_from_dataframe(master_train, encoder = encoder)
    
def cleaned_dataframes_from_dataframe(master_train, seed=1, encoder = ce.OneHotEncoder()):
    np.random.seed(seed)
    
    # copies of master
    train = pd.DataFrame.copy(master_train)
    encoded_frame = encode_categorical_data(train, encoder)
    scaled_frame = scale_continuous_data(train)
    loss = np.log(train['loss'])
    frame = pd.concat([train.ix[:, 0], loss, scaled_frame, encoded_frame], axis=1)
    return split_data(frame)
    
def select_columns(dataframe, col_type):
    headers = dataframe[:0].columns.values.T.tolist()
    col_headers = [header for header in headers if col_type in header]
    column_indicies = [i for i in range(len(headers)) if col_type in headers[i]]
    return (col_headers, dataframe[column_indicies])

def scale_continuous_data(dataframe):
    cont_headers, cont_frame = select_columns(dataframe, 'cont')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cont_frame.as_matrix())
    return pd.DataFrame(data=scaled_data, columns=cont_headers)
    
def encode_categorical_data(dataframe, encoder):
    cat_headers, cat_frame = select_columns(dataframe, 'cat')
    return encoder.fit_transform(cat_frame, None)

def split_data(dataframe):
    # building a training and validation set
    indexes = np.random.rand(len(dataframe)) < 0.8
    return (dataframe[indexes], dataframe[~indexes])

def onehot_encoder(dataframe, column):
    return pd.get_dummies(dataframe[column]).rename(columns=lambda x: column + '_' + str(x))