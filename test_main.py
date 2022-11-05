# Load Libraries for Testing
from fastapi.testclient import TestClient
from main import app

# Instantiate Test Client
client = TestClient(app)

low_salary = {"age": 39, "workclass": "State-gov", "fnlgt": 77516,
               "education": "Bachelors", "education_num": 13, "marital_status": "Never-married",
               "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White",
               "sex": "Male", "capital_gain": 2174, "capital_loss": 0,
               "hours_per_week": 40, "native_country": "United-States"
               }

high_salary = {"age": 49, "workclass": "Private",
                "fnlgt": 193524, "education": "Doctorate",
                "education_num": 16, "marital_status": "Married-civ-spouse",
                "occupation": "Prof-specialty", "relationship": "Husband",
                "race": "White", "sex": "Male", "capital_gain": 5174,
                "capital_loss": 0, "hours_per_week": 60, "native_country": "United-States"
                }


def test_index_path():
    test_response = client.get('/')
    print("testing get")
    assert test_response.status_code == 200
    assert test_response.json() == {
        "greeting": "Welcome to the API for predicting salary from census"
    }


def test_below_salary():
    test_response = client.post('/predict', json=low_salary)
    assert test_response.status_code == 200
    assert test_response.json()['result'] == "<=50K"


def test_above_salary():
    test_response = client.post('/predict', json=high_salary)
    assert test_response.status_code == 200
    assert test_response.json()['result'] == ">50K"


if __name__ == "__main__":
    test_index_path()

    test_below_salary()

    test_above_salary()
