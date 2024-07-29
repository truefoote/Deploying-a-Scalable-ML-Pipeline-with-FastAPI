import pytest
# TODO: add necessary import
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import train_test_split
from ml.data import process_data
import math
from ml.model import train_model, inference, load_model, performance_on_categorical_slice

# TODO: implement the first test. Change the function name and input as needed
def test_train_model_instance():
    """
    Tests that any model that passes through is an instance of RandomForestClassifer
    """
    # Your code here
    # Sample dataset
    data = {
        'col1' : [0, 1, 2, 3],
        'col2' : [4, 5, 6, 7], 
        'target' : [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    X = df[['col1', 'col2']]
    y = df['target']

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_inference_length():
    """
    Tests that the length of the 20% of the dataframe is the length of the prediction
    """
    # Your code here
    # Use similar code from main.py
    data_path = os.path.join("data", "census.csv")
    data = pd.read_csv(data_path)
    len_of_data = data.shape[0]
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features=cat_features, 
        label="salary",
        training=True
        )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    model = train_model(X_train, y_train)
    model_path = os.path.join("model", "model.pkl")
    model = load_model(
        model_path
    ) 
    preds = inference(model, X_test)
    assert len(preds) == (math.ceil(data.shape[0] * 0.2))


# TODO: implement the third test. Change the function name and input as needed
def test_slice_output_exists():
    """
    Tests slice_output.txt has been created
    """
    # Your code here
    assert os.path.exists('slice_output.txt')
