import pandas as pd
from flask import Flask, request, jsonify
import joblib

api = Flask(__name__)

# Load trained artifacts
best_model = joblib.load("models/rf_model_tuned.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

@api.route('/')
def home():
    return "Student Dropout and Academic Success Prediction API is running!"

@api.route('/predict', methods=['POST'])
def predict():
    """
    - If body is empty, return a default template JSON to fill.
    - If user fills template and sends back, return prediction + friendly message.
    """
    try:
        # Try to parse JSON input
        try:
            data = request.get_json(force=True)
        except Exception:
            data = {}

        # Default template
        default_template = {
            "age": 20,
            "gender": "male",           # male / female
            "tuition_paid": "yes",      # yes / no
            "sem1_passed": 5,
            "sem2_passed": 4,
            "sem1_grade": 13.0,
            "sem2_grade": 12.5,
            "scholarship": "no"         # yes / no
        }

        # If input is empty, return template
        if not isinstance(data, dict) or not data:
            return jsonify({
                "message": "Fill the template below and send it back for prediction.",
                "template": default_template
            })

        # Fill missing fields with defaults
        for key, value in default_template.items():
            if key not in data:
                data[key] = value

        # Map user-friendly input to model features
        gender_val = 1 if str(data["gender"]).lower() == "male" else 0
        tuition_val = 1 if str(data["tuition_paid"]).lower() == "yes" else 0
        scholarship_val = 1 if str(data["scholarship"]).lower() == "yes" else 0

        # NOTE:
        # For deployment simplicity, only key academic and financial inputs are collected from the user.
        # Remaining demographic and institutional features are assigned fixed representative values.
        # In a real production system, these values would be dynamically retrieved
        # from student information systems or institutional databases.

        # Full input dictionary for model
        model_input = {
            "Marital status": 1,
            "Application mode": 8,
            "Application order": 1,
            "Course": 171,
            "Daytime/evening attendance": 1,
            "Previous qualification": 1,
            "Previous qualification (grade)": 130,
            "Nationality": 1,
            "Mother's qualification": 19,
            "Father's qualification": 12,
            "Mother's occupation": 10,
            "Father's occupation": 6,
            "Admission grade": 127,
            "Displaced": 0,
            "Educational special needs": 0,
            "Debtor": 0,
            "Tuition fees up to date": tuition_val,
            "Gender": gender_val,
            "Scholarship holder": scholarship_val,
            "Age at enrollment": data["age"],
            "International": 0,
            "Curricular units 1st sem (credited)": 0,
            "Curricular units 1st sem (enrolled)": 6,
            "Curricular units 1st sem (evaluations)": 6,
            "Curricular units 1st sem (approved)": data["sem1_passed"],
            "Curricular units 1st sem (grade)": data["sem1_grade"],
            "Curricular units 1st sem (without evaluations)": 0,
            "Curricular units 2nd sem (credited)": 0,
            "Curricular units 2nd sem (enrolled)": 6,
            "Curricular units 2nd sem (evaluations)": 6,
            "Curricular units 2nd sem (approved)": data["sem2_passed"],
            "Curricular units 2nd sem (grade)": data["sem2_grade"],
            "Curricular units 2nd sem (without evaluations)": 0,
            "Unemployment rate": 7.5,
            "Inflation rate": 1.2,
            "GDP": 2.3
        }
        input_df = pd.DataFrame([model_input])
        input_df = input_df[preprocessor.feature_names_in_]
        X_processed = preprocessor.transform(input_df)

        y_pred_encoded = best_model.predict(X_processed)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)[0]

        # Friendly message
        message_map = {
            "Dropout": "High risk of dropout",
            "Enrolled": "Likely to continue studies",
            "Graduate": "Likely to graduate successfully"
        }

        return jsonify({
            "prediction": y_pred,
            "interpretation": message_map.get(y_pred, y_pred)
        })

    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500

if __name__ == "__main__":
    api.run(debug=False)

