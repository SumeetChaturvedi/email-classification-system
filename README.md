# Email Classification System

This system processes support emails by:
1. Masking PII (Personal Identifiable Information)
2. Classifying the email into categories
3. Returning both masked and demasked versions

## Features

- PII Masking: Automatically detects and masks sensitive information like:
  - Names
  - Email addresses
  - Phone numbers
  - Credit card information
  - Aadhar numbers
  - Dates of birth

- Email Classification: Categorizes emails into different types (e.g., billing, support, technical)

## How to Use

1. Enter your email text in the input box
2. Click "Submit" to process the email
3. View the results showing:
   - Original email
   - Masked version (with PII hidden)
   - Classification category
   - List of masked entities

## Example

Input:
```
Hello, my name is John Doe and my email is johndoe@example.com. I'm having trouble with my billing.
```

Output:
```
=== Email Classification Results ===

Input Email:
Hello, my name is John Doe and my email is johndoe@example.com. I'm having trouble with my billing.

Category: billing

Masked Email:
Hello, my name is [full_name] and my email is [email]. I'm having trouble with my billing.

Demasked Email:
Hello, my name is John Doe and my email is johndoe@example.com. I'm having trouble with my billing.

Masked Entities:
- full_name: John Doe
- email: johndoe@example.com
```

## Technical Details

- Built with Python and FastAPI
- Uses machine learning for classification
- Implements regex-based PII detection
- Deployed on Hugging Face Spaces

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
