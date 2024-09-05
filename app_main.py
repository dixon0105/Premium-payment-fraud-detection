from flask import Flask, request, jsonify
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
from model_opt import *

app = Flask(__name__)

high_risk_aml_countries = ['BGR','BFA','CMR','HRV',
                           'PRK','COD','HTI','IRN',
                           'KEN','MLI','MCO','MOZ',
                           'MMR','NAM','NGA','PHL',
                           'SEN','ZAF','SSD','SYR',
                           'TZA','VEN','VNM','YEM']

"""
def is_high_risk_country(bank_name):
    url = f"https://.../{...}"
    headers = {
        'Accept-Version': '3'
    }
    bank_country = requests.get(url, headers=headers)

    return bank_country in high_risk_aml_countries
"""

def is_third_party_payment(customer_name, card_holder_name):
    similarity = fuzz.ratio(card_holder_name, customer_name)
    return similarity < 80  # Threshold for considering it a third-party payment

@app.route('/fraud_check', methods=['POST'])
def fraud_check():
    try:
        data = request.json
        customer_name = data.get('customer_name')
        credit_card_number = data.get('credit_card_number')
        expiry_date = data.get('expiry_date')
        transaction_amount = data.get('transaction_amount')
        card_holder_name = data.get('card_holder_name')
        bank_name = data.get('bank_name')
        product_name = data.get('product_name')
        policy_currency = data.get('policy_currency')
        sales_channel = data.get('sales_channel')
        premium_amount = data.get('premium_amount')
        commission_amount = data.get('commission_amount')
    

        #high_risk_country = is_high_risk_country(bank_name)
        third_party_payment = is_third_party_payment(customer_name, card_holder_name)

        # Load dataset
        dataset = pd.read_csv('dummy_dataset.csv')

        # Train and optimize models
        rf_model, gb_model, xgb_model, selected_features = train_and_optimize_models(dataset)

        # Prepare features
        features = pd.DataFrame({
            'credit_card_number': [credit_card_number],
            'transaction_amount': [transaction_amount],
            'premium_amount': [premium_amount],
            'commission_amount': [commission_amount]
        })
        features['credit_card_number'] = features['credit_card_number'].astype(np.int64)
        features['transaction_amount'] = features['transaction_amount'].astype(np.float64)
        features['premium_amount'] = features['premium_amount'].astype(np.float64)
        features['commission_amount'] = features['commission_amount'].astype(np.float64)
        features = features[selected_features]
        #features = pd.DataFrame(features, columns=dataset.columns.drop('fraud_label'))[selected_features]

        # Get predictions from models
        rf_fraud_risk_score = float(rf_model.predict_proba(features)[0][1])
        gb_fraud_risk_score = float(gb_model.predict_proba(features)[0][1])
        xgb_fraud_risk_score = float(xgb_model.predict_proba(features)[0][1])

        # Avg scores for final fraud risk score
        fraud_risk_score = (rf_fraud_risk_score + gb_fraud_risk_score + xgb_fraud_risk_score) / 3

        #potential_fraud = fraud_risk_score > 0.7 or high_risk_country or third_party_payment
        potential_fraud = fraud_risk_score > 0.7 or third_party_payment

        result = {
            'rf_fraud_risk_score': rf_fraud_risk_score,
            'gb_fraud_risk_score': gb_fraud_risk_score,
            'xgb_fraud_risk_score': xgb_fraud_risk_score,
            'final_fraud_risk_score': fraud_risk_score,
            'potential_fraud': potential_fraud,
            'performance_chart': 'model_performance.png'
        }

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
