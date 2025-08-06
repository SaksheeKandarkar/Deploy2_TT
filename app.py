from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model artifacts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

def create_input_data(form_data):
    input_dict = {feature: 0 for feature in feature_names}

    # Numeric inputs
    input_dict['Price (in rupees)'] = float(form_data.get('price', 0))
    input_dict['Carpet Area in sqft'] = float(form_data.get('carpet_area', 0))
    input_dict['Super Area in sqft'] = float(form_data.get('super_area', 0))
    input_dict['Bathroom'] = int(form_data.get('bathroom', 1))
    input_dict['Balcony'] = int(form_data.get('balcony', 0))
    input_dict['BHK'] = float(form_data.get('bhk', 1))

    # Mapping dictionaries
    status_map = {'Under Construction': 0, 'Ready to Move': 1}
    transaction_map = {'Resale': 0, 'New Property': 1}
    furnishing_map = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Furnished': 2}
    facing_map = {
        'East': 3,
        'West': 2,
        'North': 2,
        'South': 4,
        'North - East': 5,
        'North - West': 5,
        'South - East': 0,
        'South - West': 1,
        'NA': 0
    }
    ownership_map = {'Freehold': 0, 'Leasehold': 1, 'Power Of Attorney': 2, 'Co-operative Society': 3}

    # Map string values to numeric codes
    input_dict['Status'] = status_map.get(form_data.get('status', ''), 0)
    input_dict['Transaction'] = transaction_map.get(form_data.get('transaction', ''), 0)
    input_dict['Furnishing'] = furnishing_map.get(form_data.get('furnishing', ''), 0)
    input_dict['facing'] = facing_map.get(form_data.get('facing', ''), 0)
    input_dict['Ownership'] = ownership_map.get(form_data.get('ownership', ''), 0)

    # Location one-hot encoding
    location = form_data.get('location', '').strip().lower().replace(' ', '-')
    location_feature = f'location_{location}'
    if location_feature in input_dict:
        input_dict[location_feature] = 1

    return input_dict

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_dict = create_input_data(request.form)
        df = pd.DataFrame([input_dict])
        df = df[feature_names]  # Ensure proper column order

        # Scale features
        scaled_features = scaler.transform(df)

        # Predict
        prediction = model.predict(scaled_features)

        return render_template('index.html', prediction=round(prediction[0], 2))

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
