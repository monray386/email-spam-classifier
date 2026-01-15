# Load model and scaler
from email_feature_extractor import EmailFeatureExtractor
import joblib

xgb_model  = joblib.load("model/xgb_refined_spam_model.pkl")


# Example email
email_text = """
Congratulations! You have won a $1000 gift card. Click here to claim your prize now!!! 
Hurry, offer expires today. Visit www.superprizes.com and enter your details immediately!!!
"""

# Extract features
extractor = EmailFeatureExtractor(email_text)
X_single = extractor.to_dataframe()

print(extractor.to_dict())



# Predict
prediction = xgb_model.predict(X_single)[0]
prob = xgb_model.predict_proba(X_single)[0]

if prediction == 1:
    print(f"⚠️ Spam Email! Confidence: {prob[1]:.2%}")
else:
    print(f"✅ Legitimate Email. Confidence: {prob[0]:.2%}")
