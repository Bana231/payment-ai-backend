from data import transactions
from feature_extractor import extract_diagnostic_features
from ml_diagnosis import diagnose


# 1. Convert the existing transaction data into ML features
features = extract_diagnostic_features(transactions)

# 2. Send those features to the Random Forest diagnostic model
result = diagnose(features)

# 3. Display what the model received
print("\nDiagnostic Features:")
print(features)

# 4. Display the model's diagnosis
print("\nML Diagnosis:")
print(result)