import pytest 
from main import gradio_interface, feature_cols

def test_gradio_prediction():

    # sett the value of all feature to 1.0
    sample_input =  {f: 1.0 for f in feature_cols}

    pred = gradio_interface(**sample_input)

    # check that prediction si numeric
    assert isinstance(pred, (int, float)), "Prediction should be numeric"