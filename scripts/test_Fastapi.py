import pytest
import requests

URL = "http://127.0.0.1:8000/predict"

# Correct input
good_input = {
    "Gameweight": 2.5,
    "BGGId": 1500,
    "NumWant": 500,
    "ComAgeRec": 12,
    "BestPlayers": 4
}

# Missing field input
missing_input = {
    "Gameweight": 2.5,
    "BGGId": 1500
}

# Wrong type input
wrong_input = {
    "Gameweight": "high",
    "BGGId": "ID1500",
    "NumWant": "many",
    "ComAgeRec": "old",
    "BestPlayers": "four"
}

@pytest.mark.parametrize(
    "input_data, expect_success",
    [
        (good_input, True),
        (missing_input, False),
        (wrong_input, False),
    ]
)
def test_api_prediction(input_data, expect_success):
    resp = requests.post(URL, json=input_data)
    if expect_success:
        assert resp.status_code == 200
        assert "prediction" in resp.json()
    else:
        assert resp.status_code == 200
        assert "error" in resp.json()

# How to run them

# input in terminal : pytest -v test/

# -v = verbose output, shows each test result.

# It will run all three files automatically.

# Any failed test will show the exact reason.
