#!/usr/bin/env python3

"""
Serve board game rating model using FastAPI + Gradio.
"""

import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

# -----------------------------------------------------
# Make "src/serving" importable
# -----------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PARENT_DIR)

from serving.inference import predict_single, feature_cols

# -----------------------------------------------------
# FastAPI app
# -----------------------------------------------------
app = FastAPI(
    title="Board Game Rating Predictor",
    description="Predict board game ratings using a trained model",
    version="1.0"
)

# -----------------------------------------------------
# Pydantic schema - FIXED: GameWeight with capital W
# -----------------------------------------------------
class GameData(BaseModel):
    GameWeight: float
    BGGId: float
    NumWant: float
    ComAgeRec: float
    BestPlayers: float

# -----------------------------------------------------
# Routes
# -----------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict_game(data: GameData):
    try:
        pred = predict_single(data.model_dump())
        return {"prediction": pred}
    except Exception as e:
        return {"error": str(e)}

# -----------------------------------------------------
# Gradio UI - FIXED: inputs dict and return format
# -----------------------------------------------------
def gradio_interface(GameWeight, BGGId, NumWant, ComAgeRec, BestPlayers):
    try:
        inputs = {
            "GameWeight": GameWeight,
            "BGGId": BGGId,
            "NumWant": NumWant,
            "ComAgeRec": ComAgeRec,
            "BestPlayers": BestPlayers
        }
        pred = predict_single(inputs)
        return f"{pred:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# Friendly labels for UI
friendly_labels = {
    'GameWeight': 'Game Weight (complexity)',
    'BGGId': 'BGG ID',
    'NumWant': 'Number Wanting Game',
    'ComAgeRec': 'Community Age Recommendation',
    'BestPlayers': 'Best Player Count'
}

gr_inputs = [gr.Number(label=friendly_labels.get(f, f)) for f in feature_cols]

demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr_inputs,
    outputs=gr.Textbox(label="Predicted Rating"),
    title="🎲 Board Game Rating Predictor",
    description="Enter board game attributes to predict its rating",
    examples=[
        [2.5, 1234, 50, 12.0, 4],
        [3.2, 5678, 120, 14.0, 5],
    ],
    theme=gr.themes.Soft()
)

app = gr.mount_gradio_app(app, demo, path="/ui")

