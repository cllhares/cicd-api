# Import Libraries
import requests

# Instantiate the JSON Data
json_data = {
    "age": 39, "workclass": "State-gov", "fnlgt": 77516,
    "education": "Bachelors", "education_num": 13, "marital_status": "Never-married",
    "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White",
    "sex": "Male", "capital_gain": 2174, "capital_loss": 0,
    "hours_per_week": 40, "native_country": "United-States"
}

# Make the Request to the Live API
the_request = requests.post('https://blooming-sea-13310.herokuapp.com/predict',
                           json=json_data)

# Print the Request Reponse
print(f"The Request Response Code is {the_request.status_code} \n"
      f"The Response Body is: {the_request.json()}")
