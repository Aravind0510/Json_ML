from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model pipeline
model_pipeline = joblib.load('insurance_prediction.pkl')

def preprocess_custom_data(df):
    # Apply necessary transformations similar to the ones used during model training
    df['allergies'] = df['allergies'].apply(lambda x: ','.join(x) if isinstance(x, (list, np.ndarray)) else x)
    df['surgeries'] = df['surgeries'].apply(lambda x: ','.join(x) if isinstance(x, (list, np.ndarray)) else x)
    
    # Ensure you only drop columns that are not used in the model
    columns_to_drop = ['name']  # Adjust according to your model's needs
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, errors='ignore')
    
    return df

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    patient_id = int(request.form['patient_id'])
    age = int(request.form['age'])
    past_medication_done = (request.form['past_medication_done'])
    allergies = request.form['allergies']
    chronic_condition = request.form['chronic_condition']
    surgeries = request.form['surgeries']
    disease_condition = request.form['disease_condition']
    patient_condition = request.form['patient_condition']
    previous_insurance_claims = int(request.form['previous_insurance_claims'])
    
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'patient_id': [patient_id],
        'age': [age],
        'past_medication_done': [past_medication_done],
        'allergies': [allergies],
        'chronic_condition': [chronic_condition],
        'surgeries': [surgeries],
        'disease_condition': [disease_condition],
        'patient_condition': [patient_condition],
        'previous_insurance_claims': [previous_insurance_claims]
    })
    
    # Preprocess the input data
    input_data_preprocessed = preprocess_custom_data(input_data)
    
    # Ensure the input data is still a DataFrame
    if not isinstance(input_data_preprocessed, pd.DataFrame):
        return "Error: Input data is not a DataFrame.", 400
    
    # Make a prediction
    prediction = model_pipeline.predict(input_data_preprocessed)
    
    return render_template('index1.html', prediction_text=f'Insurance Plan is : {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)
