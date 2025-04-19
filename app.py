import os
import gradio as gr
from utils import PIIMasker
import pickle
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the PII masker
pii_masker = PIIMasker()

# Load the trained model
MODEL_DIR = "model"  # Changed back to "model" directory
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

def process_email(email_text: str) -> str:
    """
    Process a single email: mask PII, classify, and return results.
    
    Args:
        email_text: The email text to process
        
    Returns:
        Formatted results string
    """
    if email_classifier is None:
        return "Error: Model not loaded. Please ensure the model is properly trained and loaded."
    
    if not email_text or not isinstance(email_text, str):
        return "Error: Invalid email text. Please provide a non-empty string."
    
    try:
        # Mask PII in the email
        masked_email, masked_entities = pii_masker.mask_pii(email_text)
        
        # Classify the masked email
        try:
            category = email_classifier.predict([masked_email])[0]
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            return f"Error during model prediction: {str(e)}"
        
        # Demask the email
        demasked_email = pii_masker.unmask_pii(masked_email)
        
        # Format the results
        output = []
        output.append("=== Email Classification Results ===")
        output.append(f"\nInput Email:\n{email_text}")
        output.append(f"\nCategory: {category}")
        output.append(f"\nMasked Email:\n{masked_email}")
        output.append(f"\nDemasked Email:\n{demasked_email}")
        
        if masked_entities:
            output.append("\nMasked Entities:")
            for entity in masked_entities:
                output.append(f"- {entity['classification']}: {entity['entity']}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Error processing email: {str(e)}")
        return f"Error processing email: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=process_email,
    inputs=gr.Textbox(
        label="Email Text",
        placeholder="Enter your email text here...",
        lines=5
    ),
    outputs=gr.Textbox(
        label="Results",
        lines=10
    ),
    title="Email Classification System",
    description="""This system:
    1. Masks PII (Personal Identifiable Information) in emails
    2. Classifies the email into categories
    3. Returns both masked and demasked versions
    
    Try entering an email with sensitive information like names, emails, or phone numbers!""",
    examples=[
        ["Hello, my name is John Doe and my email is johndoe@example.com. I'm having trouble with my billing."],
        ["Hi, I need to reset my password. My phone number is 123-456-7890."],
        ["Please update my credit card ending in 1234. My CVV is 123."]
    ]
)

# Launch the interface
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)