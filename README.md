---
title: Email Classification System
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# Email Classification System

This is an intelligent email classification system that can:
1. Classify emails into different categories
2. Mask Personal Identifiable Information (PII)
3. Provide both masked and demasked versions of the email

## Features

- **Email Classification**: Automatically categorizes emails based on their content
- **PII Masking**: Detects and masks sensitive information like:
  - Names
  - Email addresses
  - Phone numbers
  - Credit card numbers
  - CVV numbers
  - Dates of birth
  - Aadhar numbers
- **Interactive Interface**: Easy-to-use web interface for testing the system

## How to Use

1. Enter your email text in the input box
2. Click "Submit"
3. View the results showing:
   - Original email
   - Category classification
   - Masked version (with PII hidden)
   - Demasked version
   - List of detected PII entities

## Example

Try these example inputs:
```
Hello, my name is John Doe and my email is johndoe@example.com. I'm having trouble with my billing.
```
```
Hi, I need to reset my password. My phone number is 123-456-7890.
```
```
Please update my credit card ending in 1234. My CVV is 123.
```

## Technical Details

- Built with Python and Gradio
- Uses machine learning for classification
- Implements regex-based PII detection
- Handles multiple types of sensitive information

## Model Information

The system uses a trained classifier model (`email_classifier.pkl`) that categorizes emails based on their content. The model is loaded from the `model` directory.

## Privacy
All processing is done locally in your browser. No data is stored or transmitted to external servers.

## License
MIT License

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd email-classification-system
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

To train a new model:
```bash
python app.py --mode train --data_path data/emails.csv --model_type traditional --classifier_type naive_bayes
```

### Evaluating a Model

To evaluate an existing model:
```bash
python app.py --mode evaluate --data_path data/emails.csv --model_path model/email_classifier.pkl
```

### Running the API

To start the API server:
```bash
python app.py --mode api --model_path model/email_classifier.pkl
```

The API will be available at `http://localhost:8000`

### Testing the API

To test the API:
```bash
python test.py
```

## API Endpoints

- `POST /classify_email`: Classify an email
- `GET /health`: Check API health status

## Data Format

The input CSV file should have the following columns:
- `email`: The email text
- `type`: The category of the email (for training)

## Model Types

1. Traditional ML Models:
   - Naive Bayes
   - Random Forest
   - SVM

2. Transformer Models:
   - DistilBERT (default)
   - Other Hugging Face models can be used

## PII Masking

The system masks the following types of PII:
- Full names
- Email addresses
- Phone numbers
- Dates of birth
- Aadhar numbers
- Credit/Debit card numbers
- CVV numbers
- Expiry dates
