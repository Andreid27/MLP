# Import pandas and numpy
import pandas as pd
import numpy as np

# Import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data():


    # Load the CSV file as a pandas DataFrame
    df = pd.read_csv("wnw_greece.csv")

    # Select the columns that contain the features you want
    # In this case, we assume you want all the columns except the first one (recordid) and the last one (Positive PCR)
    features = df.iloc[:, 1:-1]

    #--DATA PROCESSING --

    # Convert the date columns into datetime objects
    features["date_used_for_statistics"] = pd.to_datetime(features["date_used_for_statistics"])
    features["date_of_onset"] = pd.to_datetime(features["date_of_onset"])
    features["date_of_diagnosis"] = pd.to_datetime(features["date_of_diagnosis"])


    # Convert "place_of_Infection" column to categorical
    features["place_of_Infection"] = features["place_of_Infection"].astype("category")
    # Assign numerical codes to the categories
    features["place_of_Infection_code"] = features["place_of_Infection"].cat.codes

    # List of columns to ignore (date columns)
    date_columns_to_ignore = ["date_used_for_statistics", "date_of_onset", "date_of_diagnosis", "place_of_Infection"]

    # Drop the date columns from the DataFrame
    features = features.drop(date_columns_to_ignore, axis=1)


    # Convert the gender column into two binary columns (F and M)
    features["gender"] = features["gender"].map(gender_to_num)

    # Convert the clinical manifestation column into binary columns (NEURO and OTHER)
    features["clinical_manifestation"] = features["clinical_manifestation"].map(neuro_to_num)

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

    # Split the data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # -------- DATA FOR FRUITS TEST ---------
    fruits = pd.read_table('fruit_data_with_colors.txt')
    fruits.head()

    # Prepare data for classification

    feature_names = ['mass', 'width', 'height', 'color_score']
    X = fruits[feature_names]
    y = fruits['fruit_label']
    # X1 = fruits1[feature_names]
    # y1 = fruits1['fruit_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # # Create two new columns for each possible value of the label using pandas.get_dummies()
    label_train = pd.get_dummies(y_train, prefix="fruit_label")
    label_test = pd.get_dummies(y_test, prefix="fruit_label")

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
        return None # in case of missing or invalid values

def neuro_to_num(gender):
    if gender == "NEURO":
        return 0
    elif gender == "O":
        return 1
    else:
        return None # in case of missing or invalid values