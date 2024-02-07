"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from scipy.stats.mstats import winsorize
from sklearn import impute
from sklearn.impute import SimpleImputer


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector = pd.DataFrame.from_dict([feature_vector_dict])

    train_df= feature_vector.copy( deep=True)

    # Feature has no significance so its removed
    train_df = train_df.drop(['Unnamed: 0'], axis=1)

    # Convert feature data type to floats
    train_df['Valencia_wind_deg'] = train_df['Valencia_wind_deg'].astype(str).str.extract('(\d+)', expand=False).astype(float)
    train_df['Seville_pressure'] = train_df['Seville_pressure'].astype(str).str.extract('(\d+)', expand=False).astype(float)

    # Create a copy
    Train = train_df.copy(deep='True')

    # Split time to create new features 
    Train['time'] = pd.to_datetime(Train['time'])
    Train['Day'] = Train['time'].dt.day
    Train['Month'] = Train['time'].dt.month
    Train['Year'] = Train['time'].dt.year
    Train['Hour'] = Train['time'].dt.hour

    # Pass all features that were used from the note for during the model training phase to a new feature
    predict_features = Train[['Year', 'Month', 'Day', 'Hour', 'Madrid_wind_speed',
       'Madrid_clouds_all', 'Madrid_pressure', 'Madrid_weather_id',
       'Seville_humidity', 'Seville_clouds_all', 'Seville_wind_speed',
       'Seville_pressure', 'Seville_weather_id', 'Barcelona_wind_speed',
       'Barcelona_wind_deg', 'Barcelona_pressure', 'Barcelona_weather_id',
       'Valencia_wind_speed', 'Valencia_wind_deg', 'Valencia_humidity', 'Bilbao_wind_speed', 'Bilbao_wind_deg',
       'Bilbao_clouds_all', 'Bilbao_pressure', 'Bilbao_weather_id']]
    predict_features.info()


    return predict_features

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    imputer = SimpleImputer(strategy='mean')

    # Fit the imputer on your data and transform it
    # prep_data_imputed = imputer.fit_transform(prep_data)
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()