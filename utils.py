import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class PIIMasker:
    """Class to handle masking and unmasking of PII/PCI data."""
    
    def __init__(self):
        # Regex patterns for different PII/PCI entities
        self.patterns = {
            "full_name": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_number": r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "dob": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "aadhar_num": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "credit_debit_no": r'\b(?:\d{4}[-\s]?){4}\b',
            "cvv_no": r'\bCVV:?\s*\d{3,4}\b|\bcvv:?\s*\d{3,4}\b',
            "expiry_no": r'\b(0[1-9]|1[0-2])[/\-]\d{2,4}\b'
        }
        
        # Store masked entities for later reconstruction
        self.masked_entities = {}
        self.entity_count = {}
        
    def mask_pii(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Mask PII entities in text.
        
        Args:
            text: Input email text
            
        Returns:
            Tuple containing masked text and list of masked entities
        """
        masked_text = text
        entities_info = []
        
        # Reset for each new text
        self.masked_entities = {}
        self.entity_count = {entity_type: 0 for entity_type in self.patterns.keys()}
        
        # Process each entity type
        for entity_type, pattern in self.patterns.items():
            masked_text, entities = self._process_entity(masked_text, entity_type, pattern)
            entities_info.extend(entities)
            
        return masked_text, entities_info
    
    def _process_entity(self, text: str, entity_type: str, pattern: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Process and mask a specific entity type in the text."""
        entities_info = []
        
        # Find all matches
        for match in re.finditer(pattern, text):
            start, end = match.span()
            original_entity = text[start:end]
            
            # Create unique identifier for this entity
            self.entity_count[entity_type] += 1
            masked_id = f"{entity_type}_{self.entity_count[entity_type]}"
            
            # Store original value for reconstruction
            if entity_type not in self.masked_entities:
                self.masked_entities[entity_type] = {}
            self.masked_entities[entity_type][masked_id] = original_entity
            
            # Create mask placeholder
            mask_placeholder = f"[{entity_type}]"
            
            # Store entity information
            entities_info.append({
                "position": [start, end],
                "classification": entity_type,
                "entity": original_entity
            })
            
            # Replace in text - need to recalculate text and adjust positions for subsequent matches
            text = text[:start] + mask_placeholder + text[end:]
            
        return text, entities_info
    
    def unmask_pii(self, masked_text: str) -> str:
        """
        Restore the original PII entities in masked text.
        
        Args:
            masked_text: Text with masked PII entities
            
        Returns:
            Text with original PII entities restored
        """
        unmasked_text = masked_text
        
        for entity_type in self.patterns.keys():
            # Replace all occurrences of this entity type
            pattern = f"\\[{entity_type}\\]"
            
            matches = list(re.finditer(pattern, unmasked_text))
            for i, match in enumerate(matches):
                entity_id = f"{entity_type}_{i+1}"
                if entity_type in self.masked_entities and entity_id in self.masked_entities[entity_type]:
                    original_value = self.masked_entities[entity_type][entity_id]
                    start, end = match.span()
                    unmasked_text = unmasked_text[:start] + original_value + unmasked_text[end:]
        
        return unmasked_text

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    # Load the data
    df = pd.read_csv('/Users/sumeetchaturvedi/Downloads/emails.csv')
    
    # Create an instance of PIIMasker
    masker = PIIMasker()
    
    # Mask PII in emails
    masked_emails = []
    entities_list = []
    
    for email in df['email']:  # Changed from 'email_body' to 'email'
        masked_email, entities = masker.mask_pii(email)
        masked_emails.append(masked_email)
        entities_list.append(entities)
    
    # Add masked emails to DataFrame
    df['masked_email'] = masked_emails
    df['masked_entities'] = entities_list
    
    return df

def prepare_text_for_classification(text: str) -> str:
    """
    Prepare text for classification by cleaning and normalizing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned and normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters and numbers (optional - may want to keep some for emails)
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Training and testing DataFrames
    """
    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split point
    split_point = int(len(df) * (1 - test_size))
    
    # Split the data
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]
    
    return train_df, test_df