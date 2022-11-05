import os
import sys
import json
import joblib
import pickle
import pandas as pd
from pandas import json_normalize
from typing import Union
from fastapi import FastAPI, Query
from pydantic import BaseModel
from starter.ml.data import process_data
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

the_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def load_model(model_path):
    with open(os.path.join(model_path, "census-lr-model.pkl"), 'rb') as file:
        model_estimator = pickle.load(file)

    with open(os.path.join(model_path, "census-lr-encoder-model.pkl"), 'rb') as file:
        model_encoder = pickle.load(file)

    with open(os.path.join(model_path, "census-lr-lb-model.pkl"), 'rb') as file:
        model_lb = pickle.load(file)

    print("Model loaded \n")
    return model_estimator, model_encoder, model_lb


# Instantiate the API
app = FastAPI()

model_est, model_enc, model_l = load_model('model')

encode = pickle.load(open("./model/census-lr-encoder-model.pkl", 'rb'))


class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    # marital_status: str = Query(None, alias='marital-status')
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    # native_country: str = Query(None, alias='native-country')


@app.get("/")
async def greeting():
    return {
        "greeting": "Welcome to the API for predicting salary from census"
    }


@app.post('/predict')
async def create_item(person_item: Person):
    print(person_item)
    df = pd.DataFrame([person_item.dict()])
    print(df)
    to_predict, _, _, _ = process_data(df, cat_features, training=False, encoder=encode)

    response = model_est.predict(to_predict)

    if int(response) == 1:
        the_result = '>50K'
    else:
        the_result = '<=50K'

    print(f"the response: , {response}, {the_result}")
    return {
        "result": the_result
    }
