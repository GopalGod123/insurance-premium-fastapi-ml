from fastapi import FastAPI
from fastapi.responses import JSONResponse
from model.predict import predict_output, MODEL_VERSION, model
from Schema.user_input import UserInput
from Schema.prediction_response import PredictionResponse
import pandas as pd


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Insurance Premium Prediction API. Use the /predict endpoint to get predictions."}
@app.get("/health")
def health_check():
    return {
        'status': 'OK',
        'version': MODEL_VERSION,
        'model_loaded': model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_premium(data: UserInput):
    
    user_input = {
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'occupation': data.occupation,
        'income_lpa': data.income_lpa,
        'smoker': data.smoker,
        'age': data.age,
        'weight': data.weight,
        'height': data.height
    }

    try:
        prediction= predict_output(user_input)
        return JSONResponse(status_code=200, content={"response": prediction})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})