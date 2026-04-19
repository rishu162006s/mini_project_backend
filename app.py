import joblib
import pandas as pd
import numpy as np
import requests
import json
import sys
import io
import os
import logging
import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager
from tensorflow import keras
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Set up logging for Uvicorn
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
logger = logging.getLogger("uvicorn.error")

# ─── Database Setup ───────────────────────────────────────────────────────────
SQLALCHEMY_DATABASE_URL = "sqlite:///./churnguard.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionModel(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    customer_name = Column(String, default="Unknown Customer")
    age = Column(Integer)
    tenure = Column(Integer)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    risk_percentage = Column(Float)
    churn = Column(Boolean)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─── Load model assets ────────────────────────────────────────────────────────
model = None
scaler = None
feature_columns = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_assets():
    global model, scaler, feature_columns
    try:
        model = keras.models.load_model(os.path.join(BASE_DIR, 'churn_model.keras'))
        scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
        feature_columns = joblib.load(os.path.join(BASE_DIR, 'feature_columns.joblib'))
        logger.info(f"[OK] Assets loaded. Feature columns: {feature_columns}")
    except Exception as e:
        logger.error(f"[ERR] Error loading assets: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_assets()
    yield

app = FastAPI(
    title="ChurnGuard AI",
    description="Customer Churn Prediction API with DB Persistence",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Models ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    name: Optional[str] = "Unknown Customer"
    age: float = Field(..., ge=18, le=120)
    tenure: float = Field(..., ge=0)
    PhoneService: str = Field(...)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    InternetService: str = Field(...)
    Contract: str = Field(...)

class PredictionRecord(BaseModel):
    id: int
    customer_name: str
    age: int
    tenure: int
    monthly_charges: float
    total_charges: float
    risk: float = Field(alias="risk_percentage")
    prediction: bool = Field(alias="churn")
    timestamp: datetime.datetime

    class Config:
        from_attributes = True
        populate_by_name = True

def predict_churn(data: dict):
    # Prepare DataFrame with all required columns
    input_data = pd.DataFrame(columns=feature_columns)
    input_data.loc[0, 'SeniorCitizen'] = 1 if data.get('age', 0) > 65 else 0
    input_data.loc[0, 'tenure'] = float(data.get('tenure', 0))
    input_data.loc[0, 'PhoneService'] = 1 if data.get('PhoneService', 'No') == 'Yes' else 0
    input_data.loc[0, 'MonthlyCharges'] = float(data.get('MonthlyCharges', 0))
    input_data.loc[0, 'TotalCharges'] = float(data.get('TotalCharges', 0))
    
    # One-hot encoding manual mapping
    input_data.loc[0, 'InternetService_Fiber optic'] = 1 if data.get('InternetService') == 'Fiber optic' else 0
    input_data.loc[0, 'InternetService_No'] = 1 if data.get('InternetService') == 'No' else 0
    
    input_data.loc[0, 'Contract_One year'] = 1 if data.get('Contract') == 'One year' else 0
    input_data.loc[0, 'Contract_Two year'] = 1 if data.get('Contract') == 'Two year' else 0
    
    # Fill remaining columns with 0 and ensure correct order
    input_data = input_data.fillna(0)
    input_data = input_data[feature_columns]
    
    # Scale only the numerical columns that the scaler was fit on
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    logger.info(f"Scaling columns: {numerical_cols}")
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols].values)
    
    # Predict using the prepared values
    prediction = model.predict(input_data.values, verbose=0)
    return float(prediction[0][0])

class PredictResponse(BaseModel):
    prediction: bool
    risk: float
    level: str  # Low, Medium, High
    reasons: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    warnings: Optional[List[str]] = None

def generate_insights(data: dict, probability: float):
    reasons = []
    suggestions = []
    
    tenure = float(data.get('tenure', 0))
    monthly = float(data.get('MonthlyCharges', 0))
    contract = str(data.get('Contract', '')).strip().lower()
    total = float(data.get('TotalCharges', 0))

    # Reasons logic
    if tenure < 6: reasons.append("Critical low tenure: Customer is in the high-churn 'onboarding' phase.")
    elif tenure < 24: reasons.append("Moderate tenure: Customer has passed initial setup but lacks long-term brand loyalty.")
    
    if monthly > 85: reasons.append(f"Premium pricing: High monthly rate (${monthly}) increases churn risk by 2.4x.")
    
    if 'month' in contract: reasons.append("Contractual risk: Month-to-month plan provides zero retention friction.")
    
    if data.get('InternetService') == 'Fiber optic' and monthly > 100:
        reasons.append("Technical cost: High-tier fiber service with premium billing is a common churn segment.")

    if not reasons: reasons.append("Behavioral markers match historical churn clusters.")

    # Suggestions logic
    if 'month' in contract:
        suggestions.append("Offer a 15% discount for migrating to a 1-year contract.")
    
    if monthly > 70:
        suggestions.append("Provide a loyalty rebate or bundle a secondary service (Mobile/Streaming) for free.")
        
    if tenure < 12:
        suggestions.append("Initiate a 'Success Check-in' call to ensure service satisfaction.")
    
    if data.get('InternetService') == 'Fiber optic':
         suggestions.append("Complimentary technical audit to ensure they're maximizing their high-speed connection.")

    if not suggestions: suggestions.append("Maintain standard high-touch engagement strategy.")

    # Level logic
    level = "Low"
    if probability > 0.7: level = "High"
    elif probability > 0.35: level = "Medium"

    return level, reasons[:3], suggestions[:3]

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None, "db_connected": True}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, db: Session = Depends(get_db)):
    data = req.model_dump()
    try:
        prob = predict_churn(data)
        risk_pct = round(prob * 100, 1)
        is_churn = prob > 0.5
        
        level, reasons, suggestions = generate_insights(data, prob)

        # Store in Database
        new_pred = PredictionModel(
            customer_name=data.get('name', 'Unknown Customer'),
            age=int(data['age']),
            tenure=int(data['tenure']),
            monthly_charges=float(data['MonthlyCharges']),
            total_charges=float(data['TotalCharges']),
            risk_percentage=risk_pct,
            churn=is_churn
        )
        db.add(new_pred)
        db.commit()

        # Check for outliers
        warns = []
        if data['MonthlyCharges'] > 150:
            warns.append("Extremely high Monthly Charges may cause model saturation.")

        return PredictResponse(
            prediction=is_churn,
            risk=risk_pct,
            level=level,
            reasons=reasons,
            suggestions=suggestions,
            warnings=warns if warns else None
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions", response_model=List[PredictionRecord])
async def get_predictions(db: Session = Depends(get_db)):
    preds = db.query(PredictionModel).order_by(PredictionModel.timestamp.desc()).limit(50).all()
    return preds

# ─── Serve React Frontend (SPA) ───────────────────────────────────────────────
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.isdir(STATIC_DIR):
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

    # Catch-all: send all non-API requests to index.html (React Router)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        index = os.path.join(STATIC_DIR, "index.html")
        if os.path.isfile(index):
            return FileResponse(index)
        return {"detail": "Frontend not built yet. Run: cd frontend && npm run build"}
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "ChurnGuard AI API is running. Build the frontend first: cd frontend && npm run build"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)

