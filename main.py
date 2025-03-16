from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Allow all origins (or you can specify certain origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" means allowing all origins, replace with specific URLs if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Load Models
MODEL_PATHS = {
    "DT": "Models/dt.pkl",
    "RF": "Models/rf.pkl",
    "LGBM": "Models/lgbm.pkl",
}

models = {algo: pickle.load(open(path, "rb")) for algo, path in MODEL_PATHS.items()}


# Request model
class ForexInput(BaseModel):
    time: str  # Format: YYYY-MM-DD HH:MM:SS
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    model_choice: str  # "DT", "RF", "LGBM"

# Feature Engineering function
def feature_engineering(data: dict):
    df = pd.DataFrame([data])

    # ============ TIME FEATURES ============✅
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute
    df["day_of_week"] = df["time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ============ CANDLE SHAPE FEATURES ============✅
    df['candle_body'] = df['close'] - df['open']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['candle_range'] = df['high'] - df['low']

    df.drop(columns=["time"], inplace=True)

    return df

@app.post("/predict")
def predict(data: ForexInput):
    # Convert input to DataFrame
    df = feature_engineering(data.dict())
    # Select models based on user choice
    model_choice = data.model_choice
    if model_choice not in models:
        return {"error": "Invalid model choice. Choose from: DT, RF, LGBM"}
    
    model = models[model_choice]

    df.drop(columns=["model_choice"], inplace=True)

    # Make predictions
    pred = model.predict(df)[0]

    return {"predicted_close": pred}

# Run the app with: uvicorn main:app --reload
