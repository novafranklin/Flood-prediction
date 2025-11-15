from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import pandas as pd
import io
import json

from model import FloodPredictionModel, generate_sample_dataset

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global model instance
flood_model = FloodPredictionModel()
current_dataset = None

# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class PredictionInput(BaseModel):
    rainfall: float
    temperature: float
    humidity: float
    pressure: float
    model_type: str = 'random_forest'

class TrainingResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[dict] = None

# Routes
@api_router.get("/")
async def root():
    return {"message": "Smart Flood Risk Prediction System API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks

@api_router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and preview dataset"""
    global current_dataset
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        current_dataset = df
        
        # Basic statistics
        stats = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'preview': df.head(10).to_dict('records'),
            'describe': df.describe().to_dict()
        }
        
        return JSONResponse(content={
            'success': True,
            'message': 'Dataset uploaded successfully',
            'stats': stats
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/generate-sample-dataset")
async def generate_sample():
    """Generate a sample dataset"""
    global current_dataset
    
    try:
        df = generate_sample_dataset(1000)
        current_dataset = df
        
        # Save to file
        dataset_path = ROOT_DIR / 'Flood_Prediction.csv'
        df.to_csv(dataset_path, index=False)
        
        stats = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'preview': df.head(10).to_dict('records'),
            'describe': df.describe().to_dict()
        }
        
        return JSONResponse(content={
            'success': True,
            'message': 'Sample dataset generated successfully',
            'stats': stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/train-model")
async def train_model():
    """Train both Random Forest and Logistic Regression models"""
    global current_dataset, flood_model
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload or generate a dataset first.")
    
    try:
        # Preprocess data
        flood_model.preprocess_data(current_dataset)
        
        # Train models
        flood_model.train_models()
        
        # Evaluate models
        metrics = flood_model.evaluate_models()
        
        # Save models
        flood_model.save_models()
        
        return JSONResponse(content={
            'success': True,
            'message': 'Models trained successfully',
            'metrics': metrics
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/predict")
async def predict_flood(input: PredictionInput):
    """Predict flood risk based on input parameters"""
    global flood_model
    
    # Try to load models if not already loaded
    if flood_model.rf_model is None:
        loaded = flood_model.load_models()
        if not loaded:
            raise HTTPException(status_code=400, detail="Models not trained. Please train models first.")
    
    try:
        features = [input.rainfall, input.temperature, input.humidity, input.pressure]
        result = flood_model.predict(features, input.model_type)
        
        return JSONResponse(content={
            'success': True,
            'prediction': result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/model-metrics")
async def get_model_metrics():
    """Get current model metrics"""
    global flood_model
    
    if flood_model.rf_model is None:
        loaded = flood_model.load_models()
        if not loaded:
            raise HTTPException(status_code=400, detail="Models not trained yet.")
    
    try:
        metrics = flood_model.evaluate_models()
        return JSONResponse(content={
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/dataset-visualizations")
async def get_visualizations():
    """Get data for visualizations"""
    global current_dataset
    
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    
    try:
        # Rainfall vs Flood Occurrence
        flood_data = current_dataset[current_dataset['FloodOccurrence'] == 1]
        no_flood_data = current_dataset[current_dataset['FloodOccurrence'] == 0]
        
        viz_data = {
            'rainfall_distribution': {
                'flood': flood_data['Rainfall'].tolist(),
                'no_flood': no_flood_data['Rainfall'].tolist()
            },
            'feature_averages': {
                'flood': {
                    'Rainfall': float(flood_data['Rainfall'].mean()),
                    'Temperature': float(flood_data['Temperature'].mean()),
                    'Humidity': float(flood_data['Humidity'].mean()),
                    'Pressure': float(flood_data['Pressure'].mean())
                },
                'no_flood': {
                    'Rainfall': float(no_flood_data['Rainfall'].mean()),
                    'Temperature': float(no_flood_data['Temperature'].mean()),
                    'Humidity': float(no_flood_data['Humidity'].mean()),
                    'Pressure': float(no_flood_data['Pressure'].mean())
                }
            },
            'correlation_data': current_dataset[['Rainfall', 'Temperature', 'Humidity', 'Pressure', 'FloodOccurrence']].corr().to_dict()
        }
        
        return JSONResponse(content={
            'success': True,
            'data': viz_data
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
