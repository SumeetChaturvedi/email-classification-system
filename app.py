import os
import argparse
import pandas as pd
from typing import Dict, Any
import pickle
import gradio as gr
from utils import load_and_preprocess_data, split_data, PIIMasker
from models import train_model, evaluate_model, EmailClassifier
from api import app, load_model

# Initialize the PII masker
pii_masker = PIIMasker()

# Load the trained model
model_path = "model/email_classifier.pkl"
try:
    with open(model_path, 'rb') as f:
        email_classifier = pickle.load(f)
except FileNotFoundError:
    email_classifier = None
    print(f"Warning: Model file {model_path} not found. Please train the model first.")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Email Classification System")
    parser.add_argument("--mode", choices=["train", "evaluate", "api"], default="api",
                        help="Mode to run the application in")
    parser.add_argument("--data_path", type=str, default="data/emails.csv",
                    help="Path to the dataset")
    parser.add_argument("--model_type", choices=["traditional", "transformer"], default="traditional",
                        help="Type of model to use")
    parser.add_argument("--classifier_type", choices=["naive_bayes", "random_forest", "svm"], 
                        default="naive_bayes", help="Type of classifier for traditional models")
    parser.add_argument("--model_path", type=str, default="model/email_classifier.pkl",
                        help="Path to save/load the model")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use for testing")
    
    return parser.parse_args()

def train(args) -> Dict[str, Any]:
    """
    Train the email classification model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with training results
    """
    print(f"Loading data from {args.data_path}...")
    df = load_and_preprocess_data(args.data_path)
    
    print(f"Splitting data with test_size={args.test_size}...")
    train_df, test_df = split_data(df, test_size=args.test_size)
    
    print(f"Training {args.model_type} model with {args.classifier_type} classifier...")
    model = train_model(train_df, model_type=args.model_type, classifier_type=args.classifier_type)
    
    # Save the model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save_model(args.model_path)
    print(f"Model saved to {args.model_path}")
    
    # Evaluate on test data
    print("Evaluating model on test data...")
    results = evaluate_model(model, test_df)
    
    return {
        "model": model,
        "results": results,
        "train_size": len(train_df),
        "test_size": len(test_df)
    }

def evaluate(args) -> Dict[str, Any]:
    """
    Evaluate a trained model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Loading data from {args.data_path}...")
    df = load_and_preprocess_data(args.data_path)
    
    print(f"Loading model from {args.model_path}...")
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Evaluating model...")
    results = evaluate_model(model, df)
    
    return {
        "results": results,
        "data_size": len(df)
    }

def start_api(args) -> None:
    """
    Start the API server.
    
    Args:
        args: Command line arguments
    """
    from api import start_api as start_api_server, load_model
    
    print(f"Loading model from {args.model_path}...")
    try:
        load_model(args.model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Starting API server...")
    try:
        start_api_server(host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Error starting API server: {e}")
        return

def process_email(email_text: str) -> Dict[str, Any]:
    """
    Process a single email: mask PII, classify, and return results.
    
    Args:
        email_text: The email text to process
        
    Returns:
        Dictionary containing processing results
    """
    if email_classifier is None:
        return {
            "error": "Model not loaded. Please ensure the model is properly trained and loaded."
        }
    
    try:
        # Mask PII in the email
        masked_email, masked_entities = pii_masker.mask_pii(email_text)
        
        # Classify the masked email
        category = email_classifier.predict([masked_email])[0]
        
        # Demask the email
        demasked_email = pii_masker.unmask_pii(masked_email)
        
        # Prepare response
        return {
            "input_email": email_text,
            "masked_email": masked_email,
            "demasked_email": demasked_email,
            "category": category,
            "masked_entities": masked_entities
        }
        
    except Exception as e:
        return {
            "error": f"Error processing email: {str(e)}"
        }

def format_results(results: Dict[str, Any]) -> str:
    """Format the results for display."""
    if "error" in results:
        return f"Error: {results['error']}"
    
    output = []
    output.append("=== Email Classification Results ===")
    output.append(f"\nInput Email:\n{results['input_email']}")
    output.append(f"\nCategory: {results['category']}")
    output.append(f"\nMasked Email:\n{results['masked_email']}")
    output.append(f"\nDemasked Email:\n{results['demasked_email']}")
    
    if results['masked_entities']:
        output.append("\nMasked Entities:")
        for entity in results['masked_entities']:
            output.append(f"- {entity['classification']}: {entity['entity']}")
    
    return "\n".join(output)

def process_and_display(email_text: str) -> str:
    """Process the email and return formatted results."""
    results = process_email(email_text)
    return format_results(results)

# Create Gradio interface
demo = gr.Interface(
    fn=process_and_display,
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

def main():
    """Main function to run the application."""
    args = parse_arguments()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "api":
        start_api(args)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
    demo.launch()