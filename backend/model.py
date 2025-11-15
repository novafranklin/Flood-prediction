import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import os
from pathlib import Path
import json

ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

class FloodPredictionModel:
    def __init__(self):
        self.rf_model = None
        self.lr_model = None
        self.scaler = None
        self.feature_columns = ['Rainfall', 'Temperature', 'Humidity', 'Pressure']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Separate features and target
        X = df[self.feature_columns]
        y = df['FloodOccurrence']
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train both Random Forest and Logistic Regression models"""
        # Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Logistic Regression
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.lr_model.fit(self.X_train, self.y_train)
        
    def evaluate_models(self):
        """Evaluate both models and return metrics"""
        metrics = {}
        
        # Random Forest metrics
        rf_pred = self.rf_model.predict(self.X_test)
        rf_pred_proba = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        metrics['random_forest'] = {
            'accuracy': float(accuracy_score(self.y_test, rf_pred)),
            'precision': float(precision_score(self.y_test, rf_pred)),
            'recall': float(recall_score(self.y_test, rf_pred)),
            'f1_score': float(f1_score(self.y_test, rf_pred)),
            'confusion_matrix': confusion_matrix(self.y_test, rf_pred).tolist(),
        }
        
        # ROC curve for Random Forest
        fpr_rf, tpr_rf, _ = roc_curve(self.y_test, rf_pred_proba)
        metrics['random_forest']['roc_auc'] = float(auc(fpr_rf, tpr_rf))
        metrics['random_forest']['roc_curve'] = {
            'fpr': fpr_rf.tolist(),
            'tpr': tpr_rf.tolist()
        }
        
        # Logistic Regression metrics
        lr_pred = self.lr_model.predict(self.X_test)
        lr_pred_proba = self.lr_model.predict_proba(self.X_test)[:, 1]
        
        metrics['logistic_regression'] = {
            'accuracy': float(accuracy_score(self.y_test, lr_pred)),
            'precision': float(precision_score(self.y_test, lr_pred)),
            'recall': float(recall_score(self.y_test, lr_pred)),
            'f1_score': float(f1_score(self.y_test, lr_pred)),
            'confusion_matrix': confusion_matrix(self.y_test, lr_pred).tolist(),
        }
        
        # ROC curve for Logistic Regression
        fpr_lr, tpr_lr, _ = roc_curve(self.y_test, lr_pred_proba)
        metrics['logistic_regression']['roc_auc'] = float(auc(fpr_lr, tpr_lr))
        metrics['logistic_regression']['roc_curve'] = {
            'fpr': fpr_lr.tolist(),
            'tpr': tpr_lr.tolist()
        }
        
        return metrics
    
    def predict(self, features, model_type='random_forest'):
        """Predict flood risk for given features"""
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Choose model
        model = self.rf_model if model_type == 'random_forest' else self.lr_model
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] * 100
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High Flood Risk ⚠️' if probability > 70 else 'Low Flood Risk ✅'
        }
    
    def save_models(self):
        """Save trained models and scaler"""
        joblib.dump(self.rf_model, MODELS_DIR / 'random_forest_model.pkl')
        joblib.dump(self.lr_model, MODELS_DIR / 'logistic_regression_model.pkl')
        joblib.dump(self.scaler, MODELS_DIR / 'scaler.pkl')
        
    def load_models(self):
        """Load trained models and scaler"""
        try:
            self.rf_model = joblib.load(MODELS_DIR / 'random_forest_model.pkl')
            self.lr_model = joblib.load(MODELS_DIR / 'logistic_regression_model.pkl')
            self.scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
            return True
        except:
            return False

def generate_sample_dataset(num_samples=1000):
    """Generate a sample flood prediction dataset"""
    np.random.seed(42)
    
    # Generate features
    rainfall = np.random.uniform(0, 300, num_samples)
    temperature = np.random.uniform(15, 40, num_samples)
    humidity = np.random.uniform(30, 100, num_samples)
    pressure = np.random.uniform(980, 1030, num_samples)
    
    # Generate target with correlation to features
    # Higher rainfall, higher humidity, lower pressure -> higher flood risk
    flood_score = (
        (rainfall / 300) * 0.5 + 
        (humidity / 100) * 0.3 + 
        ((1030 - pressure) / 50) * 0.2
    )
    
    # Add some randomness
    flood_score += np.random.uniform(-0.2, 0.2, num_samples)
    
    # Convert to binary (1 if flood, 0 if no flood)
    flood_occurrence = (flood_score > 0.5).astype(int)
    
    # Create dataframe
    df = pd.DataFrame({
        'Rainfall': rainfall,
        'Temperature': temperature,
        'Humidity': humidity,
        'Pressure': pressure,
        'FloodOccurrence': flood_occurrence
    })
    
    return df
