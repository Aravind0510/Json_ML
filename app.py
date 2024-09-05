from flask import Flask, request, render_template, session
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session storage

# Load the trained model pipeline
model_pipeline = joblib.load('medication_prediction.pkl')

# Define the preprocessing function
def preprocess_data(df):
    df['allergies'] = df['allergies'].apply(lambda x: ','.join(x) if isinstance(x, (list, pd.np.ndarray)) else x)
    df['surgeries'] = df['surgeries'].apply(lambda x: ','.join(x) if isinstance(x, (list, pd.np.ndarray)) else x)
    columns_to_drop = ['location', 'name', 'insurance_plan', 'doctor_availability', 'bed_availability', 'nurse_availability']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
    return df

@app.route('/')
def home():
    # Clear session data on page load (to reset prediction)
    session.clear()
    return render_template('index.html')

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
    })
    
    # Preprocess the input data
    input_data_preprocessed = preprocess_data(input_data)
    
    # Make a prediction
    prediction = model_pipeline.predict(input_data_preprocessed)

    # Define the list of medications for which to consult Dr. John
    medications_to_consult_dr_john = ["Metformin", "Insulin", "ACE inhibitors or ARBs", "Statins", "Aspirin"]
    medications_to_consult_dr_kumar = ["ACE inhibitors", "Beta-blockers", "Diuretics", "Calcium channel blockers", "ARBs"]
    medications_to_consult_dr_ram =["Inhaled corticosteroids", "Long-acting beta-agonists", "Anticholinergics", "Phosphodiesterase-4 inhibitors", "Mucolytics"]
    medications_to_consult_dr_madhu =["Methotrexate", "Biologics (e.g., TNF inhibitors)", "NSAIDs", "Corticosteroids", "DMARDs"]
    medications_to_consult_dr_prakash = ["Statins", "Beta-blockers", "ACE inhibitors", "Antiplatelet agents (e.g., aspirin, clopidogrel)", "Diuretics"]
    # Define logic for doctor suggestion
    if prediction[0] in medications_to_consult_dr_john:
        doctor_name = "Dr. John"
        cabin_number = "12"
        doctor_recommendation = f"Please visit {doctor_name} in cabin #{cabin_number}."
    elif prediction[0] in medications_to_consult_dr_kumar:
        doctor_name = "Dr. Kumar"
        cabin_number = "05"
        doctor_recommendation = f"Please visit {doctor_name} in cabin #{cabin_number}."
    elif prediction[0] in medications_to_consult_dr_ram:
        doctor_name = "Dr.Ram"
        cabin_number = "08"
        doctor_recommendation = f"Please visit {doctor_name} in cabin #{cabin_number}."
    elif prediction[0] in medications_to_consult_dr_prakash:
        doctor_name = "Dr.Prakash"
        cabin_number = "10"
        doctor_recommendation = f"Please visit {doctor_name} in cabin #{cabin_number}."
    elif prediction[0] in medications_to_consult_dr_madhu:
        doctor_name = "Dr.Madhu"
        cabin_number = "08"
        doctor_recommendation = f"Please visit {doctor_name} in cabin #{cabin_number}."
    else:
        doctor_recommendation = "No immediate need to visit the doctor."

    # Store the result in session
    session['prediction'] = prediction[0]
    session['doctor_name'] = doctor_name if prediction[0] in medications_to_consult_dr_john else None
    session['cabin_number'] = cabin_number if prediction[0] in medications_to_consult_dr_john else None
    session['doctor_recommendation'] = doctor_recommendation
    
    return render_template('index.html', prediction_text=f'Predicted Medication: {prediction[0]}', 
                           doctor_recommendation=doctor_recommendation)

if __name__ == '__main__':
    app.run(debug=True)
