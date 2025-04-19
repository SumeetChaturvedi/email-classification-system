import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, Tuple, List
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

class EmailClassifier:
    """Base class for email classification models."""
    
    def __init__(self, model_type: str = "traditional"):
        """
        Initialize the email classifier.
        
        Args:
            model_type: Type of model to use ("traditional" or "transformer")
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = None
        self.classes = None
    
    def train(self, X_train: List[str], y_train: List[str]) -> None:
        """
        Train the classification model.
        
        Args:
            X_train: List of masked email texts
            y_train: List of email categories
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, X: List[str]) -> List[str]:
        """
        Predict categories for new emails.
        
        Args:
            X: List of masked email texts
            
        Returns:
            Predicted categories
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def evaluate(self, X_test: List[str], y_test: List[str]) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: List of masked email texts
            y_test: List of true categories
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            "accuracy": accuracy,
            "classification_report": report
        }
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            file_path: Path to save the model
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, file_path: str) -> 'EmailClassifier':
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model

class TraditionalMLClassifier(EmailClassifier):
    """Traditional ML classifiers for email classification."""
    
    def __init__(self, classifier_type: str = "naive_bayes"):
        """
        Initialize a traditional ML classifier.
        
        Args:
            classifier_type: Type of classifier to use ("naive_bayes", "random_forest", "svm")
        """
        super().__init__("traditional")
        self.classifier_type = classifier_type
        
        # Initialize the classifier
        if classifier_type == "naive_bayes":
            classifier = MultinomialNB()
        elif classifier_type == "random_forest":
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_type == "svm":
            classifier = LinearSVC(random_state=42)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        # Create a pipeline with TF-IDF vectorizer and classifier
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', classifier)
        ])
        
        self.classes = None
    
    def train(self, X_train: List[str], y_train: List[str]) -> None:
        """
        Train the traditional ML model.
        
        Args:
            X_train: List of masked email texts
            y_train: List of email categories
        """
        # Store classes for prediction
        self.classes = list(set(y_train))
        
        # Train the model
        self.model.fit(X_train, y_train)
    
    def predict(self, X: List[str]) -> List[str]:
        """
        Predict categories for new emails.
        
        Args:
            X: List of masked email texts
            
        Returns:
            Predicted categories
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self.model.predict(X)

class TransformerClassifier(EmailClassifier):
    """Transformer-based classifier for email classification."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = None):
        """
        Initialize a transformer-based classifier.
        
        Args:
            model_name: Pretrained model name
            num_labels: Number of classification labels
        """
        super().__init__("transformer")
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = None
        self.label2id = None
    
    def _initialize_model(self, labels: List[str]) -> None:
        """Initialize the model with the correct number of labels."""
        self.num_labels = len(labels)
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)
    
    def train(self, X_train: List[str], y_train: List[str]) -> None:
        """
        Train the transformer model.
        
        Args:
            X_train: List of masked email texts
            y_train: List of email categories
        """
        # Get unique labels
        unique_labels = sorted(list(set(y_train)))
        
        # Initialize model with correct number of labels
        self._initialize_model(unique_labels)
        
        # Prepare dataset
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True)
        train_labels = [self.label2id[label] for label in y_train]
        
        class EmailDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
        
        train_dataset = EmailDataset(train_encodings, train_labels)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )
        
        # Create data collator for padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
    
    def predict(self, X: List[str]) -> List[str]:
        """
        Predict categories for new emails.
        
        Args:
            X: List of masked email texts
            
        Returns:
            Predicted categories
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model has not been trained yet.")
        
        # Tokenize inputs
        inputs = self.tokenizer(X, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
        
        # Convert prediction indices to labels
        predicted_labels = [self.id2label[p.item()] for p in predictions]
        
        return predicted_labels
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            file_path: Path to save the model
        """
        # Save model and tokenizer
        model_path = f"{file_path}_model"
        tokenizer_path = f"{file_path}_tokenizer"
        
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save metadata
        metadata = {
            "id2label": self.id2label,
            "label2id": self.label2id,
            "model_name": self.model_name,
            "num_labels": self.num_labels
        }
        
        with open(f"{file_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load_model(cls, file_path: str) -> 'TransformerClassifier':
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        # Load metadata
        with open(f"{file_path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance with correct model name
        instance = cls(model_name=metadata["model_name"], num_labels=metadata["num_labels"])
        
        # Set metadata
        instance.id2label = metadata["id2label"]
        instance.label2id = metadata["label2id"]
        
        # Load model and tokenizer
        model_path = f"{file_path}_model"
        tokenizer_path = f"{file_path}_tokenizer"
        
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        instance.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Move model to correct device
        instance.model.to(instance.device)
        
        return instance

def train_model(train_data: pd.DataFrame, model_type: str = "traditional", classifier_type: str = "naive_bayes") -> EmailClassifier:
    """
    Train an email classification model.
    
    Args:
        train_data: Training data DataFrame
        model_type: Type of model to use ("traditional" or "transformer")
        classifier_type: Type of classifier for traditional models
        
    Returns:
        Trained classifier model
    """
    X_train = train_data['masked_email'].tolist()
    y_train = train_data['type'].tolist()
    
    if model_type == "traditional":
        classifier = TraditionalMLClassifier(classifier_type)
    elif model_type == "transformer":
        classifier = TransformerClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Training {model_type} model...")
    classifier.train(X_train, y_train)
    print("Training complete!")
    
    return classifier

def evaluate_model(model: EmailClassifier, test_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained classifier model
        test_data: Testing data DataFrame
        
    Returns:
        Dictionary with evaluation metrics
    """
    X_test = test_data['masked_email'].tolist()
    y_test = test_data['type'].tolist()
    
    results = model.evaluate(X_test, y_test)
    
    print(f"Model accuracy: {results['accuracy']:.4f}")
    print("Classification report:")
    print(classification_report(y_test, model.predict(X_test)))
    
    return results