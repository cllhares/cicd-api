# Load Libraries for Testing
from fastapi.testclient import TestClient
from main import app

# Instantiate Test Client
client = TestClient(app)

below_salary = [{"age": 39, "workclass": "State-gov", "fnlgt": 77516,
                 "education": "Bachelors", "education_num": 13,
                 "marital_status": "Never-married", "occupation": "Prof-specialty",
                 "relationship": "Not-in-family", "race": "White",
                 "sex": "Male", "capital_gain": 2174, "capital_loss": 0,
                 "hours_per_week": 40, "native_country": "United-States"
                 }]

above_salary = [{"age": 49, "workclass": "Private",
                 "fnlgt": 193524, "education": "Doctorate",
                 "education_num": 16, "marital_status": "Married-civ-spouse",
                 "occupation": "Prof-specialty", "relationship": "Husband",
                 "race": "White", "sex": "Male", "capital_gain": 5174,
                 "capital_loss": 0, "hours_per_week": 60, "native_country": "United-States"
                 }]


