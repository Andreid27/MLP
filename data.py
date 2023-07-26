import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Import sklearn
from sklearn.preprocessing import StandardScaler

def preprocess_data():


    # Load the CSV file as a pandas DataFrame
    df = pd.read_csv("wnw_greece.csv")

    # Select the columns that contain the features you want
    # In this case, we assume you want all the columns except the first one (recordid) and the last one (Positive PCR)
    features = df.iloc[:, 1:-1]

    # Convert the date columns into datetime objects
    features["date_used_for_statistics"] = pd.to_datetime(features["date_used_for_statistics"])
    features["date_of_onset"] = pd.to_datetime(features["date_of_onset"])
    features["date_of_diagnosis"] = pd.to_datetime(features["date_of_diagnosis"])

    # Convert the gender column into two binary columns (F and M)
    features = pd.get_dummies(features, columns=["gender"])

    # Convert the clinical manifestation column into binary columns (NEURO and OTHER)
    features = pd.get_dummies(features, columns=["clinical_manifestation"])

    # Normalize the age column to have zero mean and unit variance
    scaler = StandardScaler()
    features["age"] = scaler.fit_transform(features["age"].values.reshape(-1, 1))

    # Replace the Case classification column with 0 and 1 (PROB and CONF)
    features["Case classification (CONF=confirmed, PROB=probable)"] = features[
        "Case classification (CONF=confirmed, PROB=probable)"].replace({"PROB": 0, "CONF": 1})

    # Drop the Case classification column from the features DataFrame and assign it to the labels DataFrame
    labels = features["Case classification (CONF=confirmed, PROB=probable)"]
    features = features.drop("Case classification (CONF=confirmed, PROB=probable)", axis=1)

    # Convert the features and labels DataFrames into numpy arrays
    X = features.to_numpy()
    y = labels.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train,y_train,X_test,y_test

