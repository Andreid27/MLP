# Import pandas and numpy
import pandas as pd
import numpy as np

# Import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data():
    df = pd.read_csv("wnf_case_eu.csv")

    # Select the columns that contain the features you want
    # In this case, we assume you want all the columns except the first one (recordid) and the last one (Positive PCR)
    features = df.iloc[:, :]

    # --DATA PROCESSING --
    date_columns_to_ignore = ["date_used_for_statistics", "date_of_onset", "date_of_diagnosis", "place_of_infection",
                              "reportingcountry", "id"]
    # Drop the date columns from the DataFrame
    features = features.drop(date_columns_to_ignore, axis=1)

    # Convert the gender column into two binary columns (F and M)
    features["gender"] = features["gender"].map(gender_to_num)

    # Convert the clinical manifestation column into binary columns (NEURO and OTHER)
    features["clinical_manifestation"] = features["clinical_manifestation"].map(neuro_to_num)
    features["labmethod"] = features["labmethod"].map(lab_to_num)
    features["specimenwnf"] = features["specimenwnf"].map(specimen_to_num)

    # Normalize the age column to have zero mean and unit variance
    scaler = StandardScaler()
    features["age"] = scaler.fit_transform(features["age"].values.reshape(-1, 1))

    # Replace the Case classification column with 0 and 1 (PROB and CONF)
    features["case_classification"] = features[
        "case_classification"].replace({"PROB": 0, "CONF": 1})

    # Drop the Case classification column from the features DataFrame and assign it to the labels DataFrame
    labels = features["case_classification"]
    features = features.drop("case_classification", axis=1)

    print(features.columns)

    # Convert the features and labels DataFrames into numpy arrays
    X = features.to_numpy()
    y = labels.to_numpy()

    # Split the data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create two new columns for each possible value of the label using pandas.get_dummies()
    label_train = pd.get_dummies(y_train, prefix="Case")
    label_test = pd.get_dummies(y_test, prefix="Case")

    # Convert the encoded label DataFrames into numpy arrays
    y_train = label_train.to_numpy()
    y_test = label_test.to_numpy()

    return X_train,y_train,X_test,y_test


def gender_to_num(gender):
    if gender == "F":
        return 0
    elif gender == "M":
        return 1
    else:
        return 2  # in case of missing or invalid values


def neuro_to_num(gender):
    if gender == "NEURO":
        return 0
    elif gender == "O":
        return 1
    else:
        return 2  # in case of missing or invalid values


# Load the CSV file as a pandas DataFrame
def lab_to_num(method):
    if method == "SIGM":
        return 0
    else:
        return 1  #


def specimen_to_num(specimen):
    if specimen == "CSF":
        return 0
    elif specimen == "BLOOD":
        return 1
    else:
        return 2  # in case of missing or invalid values