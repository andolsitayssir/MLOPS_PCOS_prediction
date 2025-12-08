import uvicorn
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import os
import logging
from fastapi.middleware.cors import CORSMiddleware

# --- Logging Configuration ---
# Replaced print statements with structured logging for Semaine 7 requirement
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 1. Pydantic Model for Input Validation ---
class PatientData(BaseModel):
    follicle_no_l: int = Field(..., description="Follicle Number (Left)", ge=0)
    follicle_no_r: int = Field(..., description="Follicle Number (Right)", ge=0)
    skin_darkening: int = Field(..., description="Skin Darkening (1=Yes, 0=No)", ge=0, le=1)
    hair_growth: int = Field(..., description="Hair Growth (1=Yes, 0=No)", ge=0, le=1)
    weight_gain: int = Field(..., description="Weight Gain (1=Yes, 0=No)", ge=0, le=1)
    cycle: int = Field(..., description="Cycle Regularity (2=Regular, 4=Irregular)", ge=2, le=5)
    fast_food: int = Field(..., description="Fast Food Consumption (1=Yes, 0=No)", ge=0, le=1)
    pimples: int = Field(..., description="Pimples (1=Yes, 0=No)", ge=0, le=1)
    amh: float = Field(..., description="AMH (ng/mL)", ge=0.0)

# --- 2. Global State & Lifespan ---
# Efficiently load models once at startup.
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    try:
        logger.info("⚡ Loading models...")
        # Check if preprocessor exists in app/models (ideal) or src (fallback)
        if os.path.exists("app/models/preprocessing.pkl"):
            models['preprocessor'] = joblib.load("app/models/preprocessing.pkl")
        else:
             # Fallback logic
             if os.path.exists("src/preprocessing.pkl"):
                models['preprocessor'] = joblib.load("src/preprocessing.pkl")
             else:
                logger.warning("Warning: preprocessing.pkl not found in expected paths.")

        if os.path.exists("app/models/best_model.pkl"):
            models['model'] = joblib.load("app/models/best_model.pkl")
        else:
            logger.warning("Warning: best_model.pkl not found.")

        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
    yield
    # Clean up (if needed)

# --- 3. The Application ---
app = FastAPI(
    title="PCOS Prediction API",
    description="Production-ready API for PCOS diagnosis.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Endpoints ---

@app.get("/")
def root():
    return {"message": "Welcome to PCOS Prediction API", "docs": "/docs"}

@app.get("/health")
def health_check():
    """Health check endpoint for Docker/K8s."""
    if not models or 'model' not in models or 'preprocessor' not in models:
        logger.error("Health check failed: Models not loaded")
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/predict")
def predict(data: PatientData):

    if 'model' not in models or 'preprocessor' not in models:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Convert Pydantic object to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        preprocessor = models['preprocessor']

        # Access components safely
        scaler = None
        pca = None
        top_features = None

        if isinstance(preprocessor, dict):
            if 'scaler' in preprocessor and 'pca' in preprocessor:
                scaler = preprocessor['scaler']
                pca = preprocessor['pca']
                top_features = preprocessor.get('top_features', [])
            else:
                 raise KeyError(f"Expected keys 'scaler' and 'pca' in preprocessor dict. Found: {list(preprocessor.keys())}")
        else:
            raise ValueError(f"Preprocessor expected to be a dict, got {type(preprocessor)}")

        # Rename columns to match training features exactly
        column_mapping = {
            "follicle_no_r": "Follicle No. (R)",
            "follicle_no_l": "Follicle No. (L)",
            "skin_darkening": "Skin darkening (Y/N)",
            "hair_growth": "hair growth(Y/N)",
            "weight_gain": "Weight gain(Y/N)",
            "cycle": "Cycle(R/I)",
            "fast_food": "Fast food (Y/N)",
            "pimples": "Pimples(Y/N)",
            "amh": "AMH(ng/mL)"
        }
        
        input_data_mapped = input_data.rename(columns=column_mapping)

        # Ensure correct column order if top_features is available
        if top_features:
             missing_cols = [col for col in top_features if col not in input_data_mapped.columns]
             if missing_cols:
                 raise ValueError(f"Missing columns: {missing_cols}")
             input_data_mapped = input_data_mapped[top_features]
        
        scaled_data = scaler.transform(input_data_mapped)
        processed_data = pca.transform(scaled_data)

        prediction = models['model'].predict(processed_data)[0]
        probability = models['model'].predict_proba(processed_data)[0][1]

        # --- Explainability Logic ---
        explanation = []
        try:
            model = models['model']
            if hasattr(model, 'coef_'):
                w_pca = model.coef_
                V = pca.components_
                w_features = np.dot(w_pca, V)[0]
                contributions = w_features * scaled_data[0]
                
                feature_contribs = []
                for idx, col in enumerate(top_features):
                    val = contributions[idx]
                    feature_contribs.append({"feature": col, "impact": float(val)})

                feature_contribs.sort(key=lambda x: abs(x['impact']), reverse=True)
                top_contributors = feature_contribs[:4]
                explanation = top_contributors
                
        except Exception as ex:
             logger.error(f"Explainability calculation failed: {ex}")

        # Log prediction result (structured)
        logger.info(f"Prediction made: {int(prediction)} (Confidence: {probability:.2f})")

        return {
            "status": "success",
            "diagnosis": {
                "prediction": int(prediction),
                "label": "Potential PCOS Detected" if prediction == 1 else "No PCOS Detected",
                "confidence_score": float(probability)
            },
            "risk_assessment": {
                "level": "Very High" if probability > 0.8 else "High" if probability > 0.6 else "Moderate" if probability > 0.5 else "Low",
                "probability_percent": f"{probability*100:.2f}%",
                "interpretation": "Strong indication of PCOS patterns." if probability > 0.7 else "Borderline indication matching some PCOS patterns." if probability > 0.5 else "No significant PCOS patterns detected."
            },
            "explanation": {
                "description": "Top factors contributing to this result:",
                "factors": explanation
            },
            "metadata": {
                "model_version": "1.0.0",
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "disclaimer": "This tool is for screening assistance only and is NOT a substitute for professional medical diagnosis. Please consult a specialist."
        }

    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# --- 5. Debug Entry Point ---
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)