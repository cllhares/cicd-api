# Script to train machine learning model.
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import logging
import pickle

# Add the necessary imports for the starter code.
from ml.data import process_data, load_data
from ml.model import train_model, inference, compute_model_metrics, \
    load_model, model_slice_performance

logging.basicConfig(
    level=logging.INFO,
    filemode='a',
    format='%(levelname)s - %(message)s')

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
    "native-country"
]

# Prepare Train Data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Prepare Test Data
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Proces the test data with the process_data function.
# Train and save a model.
logging.info(f'Saving Model')
model = train_model(X_train, y_train)

with open('./model/census-lr-model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('./model/census-lr-encoder-model.pkl', 'wb') as f1:
    pickle.dump(encoder, f1)

with open('./model/census-lr-lb-model.pkl', 'wb') as f2:
    pickle.dump(lb, f2)


# Make Inferences on Model
predictions = inference(model, X_test)
logging.info(f'Performing Model Predictions...')
print(predictions)

# Print out Model Performance
precision, recall, f_beta = compute_model_metrics(y_test, predictions)
logging.info(f'Model Performance Results: \nPrecision: {precision:3f}, Recall: {recall:3f}, f-beta: {f_beta:3f}')

# Print Out Slice Performance
model_slice_performance("./model/", test, X_test, y_test, cat_features, True)


