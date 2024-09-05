import requests

# Define the URL for the API endpoint
url = "http://127.0.0.1:5000/fraud_check"

sample_data = {
    "customer_name": "Chan Tai Man",
    "credit_card_number": "1234567812345678",
    "expiry_date": "12/25",
    "transaction_amount": 5000,
    "card_holder_name": "Chan Tai Man",
    "bank_name": "Bank A",
    "product_name": "Term Life Insurance",
    "policy_currency": "USD",
    "sales_channel": "Online",
    "premium_amount": 100000,
    "commission_amount": 5000
}

response = requests.post(url, json=sample_data)

content_type = response.headers.get('Content-Type')

if 'application/json' in content_type:
    try:
        print(response.json())
    except requests.exceptions.JSONDecodeError:
        print("Response content is not valid JSON:", response.text)
elif 'image/png' in content_type:
    with open('model_performancepng', 'wb') as f:
        f.write(response.content)
    print("PNG file saved as 'model_performance.png'")
else:
    print("Unexpected content type:", content_type)