# Import Libraries
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from ml.data import process_data, load_data
from ml.model import train_model, inference, compute_model_metrics, \
    load_model, model_slice_performance
import logging


# Add code to load in the data.
data = pd.read_csv("./data/clean_data_census.csv")
logging.info(f'Loading Data...')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
logging.info(f'Spitting Data for Training and Testing...')

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
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


# Check to see if you receive an encoder and label binarizer
def test_process_data():
    X_tp, y_tp, enc, lb = process_data(train, categorical_features=cat_features, label="salary")
    assert type(encoder) == sklearn.preprocessing._encoders.OneHotEncoder
    assert type(lb) == sklearn.preprocessing._label.LabelBinarizer


# Test if you receive a prediction
def test_inference():
    model = train_model(X_train, y_train)
    model_predictions = inference(model, X_test)
    assert len(X_train) == len(model_predictions)


# Test if you receive model metrics
def test_compute_model_metrics():
    test_model = train_model(X_train, y_train)
    test_model_predictions = inference(test_model, X_test)
    test_precision, test_recall, test_fbeta = compute_model_metrics(y_test, test_model_predictions)
    assert 0 <= test_precision <= 1
    assert 0 <= test_recall <= 1
    assert 0 <= test_fbeta <= 1


if __name__ == "__main__":

    test_process_data()

    test_inference()

    test_compute_model_metrics()