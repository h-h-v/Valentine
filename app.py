from flask import Flask, render_template, request
import pandas as pd
import pickle
import json

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the historical spending data
historical_data = pd.read_csv('data/historical_spending.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        year = int(request.form['year'])
        # Predict gift spending percentages for the given year
        # Assuming your model expects a 2D array as input
        predicted_spending = model.predict([[year]])
        # Convert predicted_spending from 2D to 1D array
        predicted_spending = predicted_spending[0]
        # Load labels for the spending categories
        labels = historical_data.columns[1:]  # Assuming the first column is year
        # Convert numpy array to dictionary with labels and values
        predicted_spending_dict = {label: val for label, val in zip(labels, predicted_spending)}
        return render_template('result.html', year=year, predicted_spending=json.dumps(predicted_spending_dict))

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
