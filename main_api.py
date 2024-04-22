from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import pickle
import model_backend
from sklearn.metrics import mean_absolute_error
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    date: str  # Expecting a date string in 'YYYY-MM-DD' format

def save_model(model, filename="prophet_model.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

def load_model(filename="prophet_model.pkl"):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load and preprocess data
df = model_backend.load_data()
df = model_backend.preprocess_data(df)

train_df, test_df = model_backend.train_test_data_split(df, test_size=0.25)

prophet_model = model_backend.train_prophet_model(train_df)
save_model(prophet_model, 'prophet_model.pkl')

def calculate_mape(actual, predicted):
    # Ensure no division by zero
    actual, predicted = np.array(actual), np.array(predicted)
    nonzero_mask = actual != 0
    if np.any(nonzero_mask):
        return np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100
    else:
        return float('inf')
    
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict_sales(request: PredictionRequest):
    # Load the trained Prophet model
    loaded_prophet_model = load_model('prophet_model.pkl')
    # Predict for the requested date using the Prophet model
    prediction = model_backend.predict_for_date(loaded_prophet_model, request.date)
    return prediction






