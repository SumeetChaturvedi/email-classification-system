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
app = FastAPI(title="Email Classification API", 
              description="API for classifying support emails while masking PII",
              version="1.0.0")

# Define request and response models
class EmailRequest(BaseModel):
    email_body: str

class MaskedEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str

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

@app.post("/classify_email", response_model=EmailResponse)
async def classify_email(request: EmailRequest) -> Dict[str, Any]:
    """
    Classify an email while masking PII.
    
    Args:
        request: EmailRequest containing the email body
        
    Returns:
        EmailResponse with masked entities and classification
    """
    if email_classifier is None:
        raise HTTPException(status_code=503, detail="Email classifier model is not loaded")
    
    email_body = request.email_body
    
    # Mask PII in the email
    masked_email, masked_entities = pii_masker.mask_pii(email_body)
    
    # Classify the masked email
    category = email_classifier.predict([masked_email])[0]
    
    # Prepare response
    response = {
        "input_email_body": email_body,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
    
    return response

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Status message
    """
    return {"status": "healthy", "model_loaded": email_classifier is not None}

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

def start_api():
    """Start the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_api()