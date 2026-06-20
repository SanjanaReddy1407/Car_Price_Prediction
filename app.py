import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# 1. Load the trained model when the application starts
with open('car_prediction.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    # Renders the input form
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 2. Get data from the HTML form fields
        present_price = float(request.form['Present_Price'])
        driven_kms = float(request.form['Driven_kms'])
        owner = int(request.form['Owner'])
        car_age = 2026 - int(request.form['Year'])

        # Categorical choices from user
        fuel_type = request.form['Fuel_Type']
        transmission = request.form['Transmission']
        seller_type = request.form['Seller_Type']

        # 3. Recreate the One-Hot Encoded structure exactly like training
        # Default all dummy columns to 0/False
        fuel_type_Diesel = 1 if fuel_type == 'Diesel' else 0
        fuel_type_Petrol = 1 if fuel_type == 'Petrol' else 0
        seller_type_Individual = 1 if seller_type == 'Individual' else 0
        transmission_Manual = 1 if transmission == 'Manual' else 0

        # 4. Construct the complete feature array matching X_train column order
        features = [
            car_age,
            present_price,
            driven_kms,
            owner,
            fuel_type_Diesel,
            fuel_type_Petrol,
            seller_type_Individual,
            transmission_Manual,
        ]

        # Convert to numpy array and reshape for a single prediction
        final_features = np.array(features).reshape(1, -1)

        # 5. Make prediction
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        # Send the result back to the webpage
        return render_template(
            'index.html',
            prediction_text=f'Estimated Selling Price: ₹ {output} Lakhs',
        )


if __name__ == '__main__':
    app.run(debug=True)