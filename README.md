# Student Dropout and Academic Success Prediction

## Project Overview
Achieved ~77% model accuracy in predicting student outcomes using Random Forest classification. This project predicts whether a student is likely to drop out, remain enrolled, or graduate successfully based on academic performance, demographic, and financial indicators. Early identification of at-risk students enables institutions to provide timely academic and financial interventions.

## Dataset
The dataset contains academic, demographic, and socioeconomic attributes collected at the time of enrollment and during early semesters.

## Approach
- Data cleaning and preprocessing using pipelines
- Feature engineering and selection to avoid data leakage
- Model training and comparison
- Hyperparameter tuning using Random Forest
- Deployment using Flask REST API

## Model
- Algorithm: Random Forest Classifier
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

## API Usage
- Endpoint: `/predict`
- Method: POST
- Input: JSON
- Output: Predicted student outcome with interpretation

## Limitations
- Some demographic and institutional features are fixed for demonstration purposes
- Model performance may vary across institutions
- Class imbalance may affect dropout prediction recall

## Future Improvements
- Dynamic feature collection from institutional systems
- Handling class imbalance using resampling or class weights
- Model monitoring and periodic retraining

## Running the API
pip install -r requirements
python api.py
