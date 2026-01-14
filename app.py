from flask import Flask, render_template, request
import joblib
from email_feature_extractor import EmailFeatureExtractor
import pandas as pd

app = Flask(__name__)

# Load trained model (joblib .pkl)
xgb_model = joblib.load("model/xgb_refined_spam_model.pkl")  

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        if email_text:
            # Extract features
            extractor = EmailFeatureExtractor(email_text)
            X_single = extractor.to_dataframe()
            
            # Predict
            pred_label = xgb_model.predict(X_single)[0]
            pred_proba = xgb_model.predict_proba(X_single)[0]

            # Prepare display
            if pred_label == 1:
                label = "Spam Email"
                confidence = pred_proba[1]
            else:
                label = "Legitimate Email"
                confidence = pred_proba[0]

            result = {
                "label": label,
                "confidence": confidence
            }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
