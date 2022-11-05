import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    print("Starting model training...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Predict using the model
    predictions = model.predict(X)

    return predictions


def load_model(model_path):
    with open(os.path.join(model_path, "census-lr-model.pkl"), 'rb') as file:
        model_estimator = joblib.load(file)

    with open(os.path.join(model_path, "census-lr-encoder-model.pkl"), 'rb') as file:
        model_encoder = joblib.load(file)

    with open(os.path.join(model_path, "census-lr-lb-model.pkl"), 'rb') as file:
        model_lb = joblib.load(file)

    return model_estimator, model_encoder, model_lb


def model_slice_performance(model_load_path, data, X_val, y_val, features, categorical):

    model, encoder, lb = load_model(model_load_path)

    y_pred = model.predict(X_val)

    with open('./slice_output.txt', 'w') as file:
        for the_feature in features:
            file.write(f"---- {the_feature} ----\n")
            if categorical:
                for the_category in data[the_feature].unique():
                    index = data[the_feature] == the_category
                    accuracy = np.mean(y_pred[index] == y_val[index])
                    file.write(f"Feature Slice: {the_feature}:{the_category} - {accuracy}\n")
                file.write("\n\n")

