from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import json
from utils import PIIMasker
import pickle

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
model_path = "model/email_classifier.pkl"
try:
    with open(model_path, 'rb') as f:
        email_classifier = pickle.load(f)
except FileNotFoundError:
    email_classifier = None
    print(f"Warning: Model file {model_path} not found. API will not classify emails until model is loaded.")

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
            detail="Email classifier model is not loaded. Please ensure the model is properly trained and loaded."
        )
    
    try:
        # Get the email text
        email_body = request.email_body
        
        # Mask PII in the email
        masked_email, masked_entities = pii_masker.mask_pii(email_body)
        
        # Classify the masked email
        category = email_classifier.predict([masked_email])[0]
        
        # Demask the email
        demasked_email = pii_masker.unmask_pii(masked_email)
        
        # Prepare response
        response = {
            "input_email_body": email_body,
            "masked_email": masked_email,
            "demasked_email": demasked_email,
            "category": category,
            "masked_entities": masked_entities
        }
        
        return response
        
    except Exception as e:
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

def load_model(model_path: str) -> None:
    """
    Load the email classification model.
    
    Args:
        model_path: Path to the saved model
    """
    global email_classifier
    try:
        with open(model_path, 'rb') as f:
            email_classifier = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        email_classifier = None

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
    start_api()