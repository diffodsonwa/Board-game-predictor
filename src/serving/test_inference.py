import pytest
from inference import predict_single


pred = predict_single(predict_single)

def test_inference_prediction():
    sample_input = {
        "Gameweight": 2.5,
        "BGGId": 1500,
        "NumWant": 500,
        "ComAgeRec": 12,
        "BestPlayers": 4
    }

    pred = predict_single(sample_input)

    # check prediction type
    assert isinstance (pred, float), "Prediction should be float"