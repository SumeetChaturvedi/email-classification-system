# Email Classification System

This is an email classification system that:
1. Masks PII (Personal Identifiable Information) in emails
2. Classifies emails into categories
3. Returns both masked and demasked versions

## Features
- PII Detection and Masking
- Email Classification
- Real-time Processing
- User-friendly Interface

## How to Use
1. Enter your email text in the input box
2. Click "Submit" to process the email
3. View the results including:
   - Original email
   - Masked email (with PII removed)
   - Classification category
   - List of masked entities

## Examples
Try these examples:
- "Hello, my name is John Doe and my email is johndoe@example.com. I'm having trouble with my billing."
- "Hi, I need to reset my password. My phone number is 123-456-7890."
- "Please update my credit card ending in 1234. My CVV is 123."

## Privacy
All processing is done locally in your browser. No data is stored or transmitted to external servers.

## Model Information
The system uses a machine learning model trained on support email data to classify emails into categories.

## License
MIT License

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
