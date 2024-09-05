from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the model and the scaler
model = pickle.load(open('models/svm_model.pkl', 'rb'))
scaler = pickle.load(open('models/standard_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    input_data = [float(x) for x in request.form.values()]
    input_data = np.array(input_data).reshape(1, -1)
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Predict the risk of heart attack
    prediction = model.predict(scaled_data)
    
    # Map the prediction to a human-readable output
    output = 'High risk' if prediction[0] == 1 else 'Low risk'
    
    # Provide suggestions based on the prediction
    suggestions = "Please consult a doctor for a detailed analysis and personalized advice." if output == 'High risk' else "Maintain a healthy lifestyle to keep your heart healthy."
    
    return render_template('index.html', prediction_text=f'Heart attack risk: {output}', suggestion_text=suggestions)

if __name__ == "__main__":
    app.run(debug=True)

# git ko lagi
