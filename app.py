import os
import argparse
import pandas as pd
from typing import Dict, Any
import pickle

# Import custom modules
from utils import load_and_preprocess_data, split_data
from models import train_model, evaluate_model
from api import app, load_model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Email Classification System")
    parser.add_argument("--mode", choices=["train", "evaluate", "api"], default="api",
                        help="Mode to run the application in")
    parser.add_argument("--data_path", type=str, default="/Users/sumeetchaturvedi/Downloads/emails.csv",
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
    from api import start_api, load_model
    
    print(f"Loading model from {args.model_path}...")
    load_model(args.model_path)
    
    print("Starting API server...")
    start_api()

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