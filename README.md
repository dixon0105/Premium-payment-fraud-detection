# Premium Payment with Credit Card Fraud Detection API

## Overview
This API is designed to detect fraudulent premium payments made using credit cards. It leverages multiple machine learning models and various checks to determine the likelihood of a transaction being fraudulent.

## Features

### Fraud Detection Checks
High AML Risk Countries: Checks if the credit card issuer bank is based in a high Anti-Money Laundering (AML) risk country.

Third-Party Payment Detection: Uses fuzzy logic to determine if the payment is potentially a third-party transaction.

Machine Learning Models: Averages the scores from RandomForest, GradientBoosting, and XGBoost models to determine the fraud label.

## Installation

###1. Clone the repository:

```bash
git clone https://github.com/dixon0105/Premium-payment-fraud-detection.git
```

###2. Navigate to the project directory:

```bash
cd Premium-payment-fraud-detection
```

###3. Install the required dependencies

```bash
pip install -r requirements.txt
```

```

## Usage

1. Start the API server:

```bash
python app_main.py
```

2.Send a POST request to the API endpoint with the required parameters

```bash
python testing_API.py
```"# Premium-payment-fraud-detection" 
