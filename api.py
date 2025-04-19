from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import json
from utils import PIIMasker
import pickle
import os
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the model class
from models import EmailClassifier

# Initialize FastAPI app
app = FastAPI(
    title="Email Classification API",
    description="API for classifying support emails with PII masking/demasking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Define request and response models
class EmailRequest(BaseModel):
    email_body: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "email_body": "Hello, my name is John Doe and my email is johndoe@example.com. I'm having trouble with my billing."
            }
        }

class MaskedEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    masked_email: str
    demasked_email: str
    category: str
    masked_entities: List[MaskedEntity]
    
    class Config:
        json_schema_extra = {
            "example": {
                "input_email_body": "Hello, my name is John Doe and my email is johndoe@example.com. I'm having trouble with my billing.",
                "masked_email": "Hello, my name is [full_name] and my email is [email]. I'm having trouble with my billing.",
                "demasked_email": "Hello, my name is John Doe and my email is johndoe@example.com. I'm having trouble with my billing.",
                "category": "billing",
                "masked_entities": [
                    {
                        "position": [18, 26],
                        "classification": "full_name",
                        "entity": "John Doe"
                    },
                    {
                        "position": [41, 60],
                        "classification": "email",
                        "entity": "johndoe@example.com"
                    }
                ]
            }
        }

# Load the trained model
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "email_classifier.pkl")

def load_model():
    """Load the trained model with proper error handling."""
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            logger.warning(f"Created model directory: {MODEL_DIR}")
            
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return None
            
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Load the model
email_classifier = load_model()

# Initialize the PII masker
pii_masker = PIIMasker()

@app.post("/api/v1/classify", response_model=EmailResponse, tags=["Classification"])
async def classify_email(request: EmailRequest) -> Dict[str, Any]:
    """
    Classify an email while handling PII data.
    
    This endpoint:
    1. Takes an email as input
    2. Masks any PII (Personal Identifiable Information)
    3. Classifies the masked email
    4. Returns both masked and demasked versions along with the classification
    
    Args:
        request: EmailRequest containing the email body
        
    Returns:
        EmailResponse with classification and PII handling results
        
    Raises:
        HTTPException: If the model is not loaded or other processing errors occur
    """
    if email_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable. Model not loaded. Please ensure the model is properly trained and loaded."
        )
    
    if not request.email_body or not isinstance(request.email_body, str):
        raise HTTPException(
            status_code=400,
            detail="Invalid email body. Please provide a non-empty string."
        )
    
    try:
        # Mask PII in the email
        masked_email, masked_entities = pii_masker.mask_pii(request.email_body)
        
        # Classify the masked email
        try:
            category = email_classifier.predict([masked_email])[0]
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during model prediction: {str(e)}"
            )
        
        # Demask the email
        demasked_email = pii_masker.unmask_pii(masked_email)
        
        return {
            "input_email_body": request.email_body,
            "masked_email": masked_email,
            "demasked_email": demasked_email,
            "category": category,
            "masked_entities": masked_entities
        }
        
    except Exception as e:
        logger.error(f"Error processing email: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing email: {str(e)}"
        )

@app.get("/api/v1/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Check the health status of the API and its components.
    
    Returns:
        Dictionary containing:
        - API status
        - Model status
        - PII masker status
    """
    return {
        "status": "healthy",
        "model_loaded": email_classifier is not None,
        "components": {
            "api": "healthy",
            "model": "healthy" if email_classifier is not None else "not_loaded",
            "pii_masker": "healthy"
        },
        "version": "1.0.0"
    }

def start_api(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
    """
    try:
        print(f"Starting API server on http://{host}:{port}")
        print("API Documentation available at:")
        print(f"- Swagger UI: http://{host}:{port}/docs")
        print(f"- ReDoc: http://{host}:{port}/redoc")
        
        uvicorn.run(
            "api:app",
            host=host,
            port=port,
            log_level="info",
            reload=True
        )
    except Exception as e:
        print(f"Error starting API server: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the Email Classification API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    args = parser.parse_args()
    
    start_api(host=args.host, port=args.port)