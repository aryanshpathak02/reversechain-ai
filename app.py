from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

model = None
label_encoders = {}
training_data = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/ecommerce')
def ecommerce():
    return render_template('ecommerce.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV or Excel files'}), 400
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        required_columns = ['Order ID', 'Item Name', 'Dispatch Location', 'Delivery Location', 'Return or Not']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Missing required columns: {missing_columns}'}), 400
        
        if len(df) < 50:
            return jsonify({'error': f'Insufficient data. Minimum 50 orders required, but got {len(df)}'}), 400
        
        valid_returns = ['Yes', 'No']
        invalid_returns = df[~df['Return or Not'].isin(valid_returns)]
        if not invalid_returns.empty:
            invalid_values = invalid_returns['Return or Not'].unique().tolist()
            return jsonify({'error': f'Invalid return values: {invalid_values}. Only "Yes" or "No" allowed'}), 400
        
        training_result = train_model(df)
        
        return jsonify({
            'success': 'File uploaded and model trained successfully',
            'rows': len(df),
            'model_accuracy': training_result['accuracy'],
            'return_rate': training_result['return_rate'],
            'features_importance': training_result['features_importance'],
            'analytics': training_result['analytics'],
            'top_products': training_result['top_products'],
            'location_insights': training_result['location_insights']
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def train_model(df):
    global model, label_encoders
    
    features = df[['Item Name', 'Dispatch Location', 'Delivery Location']]
    target = df['Return or Not']
    
    label_encoders = {}
    for column in features.columns:
        le = LabelEncoder()
        features[column] = le.fit_transform(features[column].astype(str))
        label_encoders[column] = le
    
    target_encoder = LabelEncoder()
    target_encoded = target_encoder.fit_transform(target)
    label_encoders['target'] = target_encoder
    
    X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return_rate = (df['Return or Not'] == 'Yes').mean() * 100
    feature_importance = dict(zip(features.columns, model.feature_importances_))
    
    analytics = {
        'total_orders': len(df),
        'return_orders': len(df[df['Return or Not'] == 'Yes']),
        'non_return_orders': len(df[df['Return or Not'] == 'No']),
        'unique_products': df['Item Name'].nunique(),
        'unique_locations': pd.concat([df['Dispatch Location'], df['Delivery Location']]).nunique()
    }
    
    product_returns = df.groupby('Item Name')['Return or Not'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    top_products = {
        'high_return_products': product_returns.head(5).to_dict(),
        'low_return_products': product_returns.tail(5).to_dict()
    }
    
    location_analysis = df.groupby(['Dispatch Location', 'Delivery Location'])['Return or Not'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).sort_values(ascending=False)
    
    high_risk_routes = {}
    for (dispatch, delivery), rate in location_analysis.head(5).items():
        route_key = f"{dispatch} → {delivery}"
        high_risk_routes[route_key] = float(rate)
    
    low_risk_routes = {}
    for (dispatch, delivery), rate in location_analysis.tail(5).items():
        route_key = f"{dispatch} → {delivery}"
        low_risk_routes[route_key] = float(rate)
    
    location_insights = {
        'high_risk_routes': high_risk_routes,
        'low_risk_routes': low_risk_routes
    }
    
    return {
        'accuracy': round(accuracy, 4),
        'return_rate': round(return_rate, 2),
        'features_importance': feature_importance,
        'analytics': analytics,
        'top_products': top_products,
        'location_insights': location_insights
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not trained yet. Please upload training data first.'}), 400
        
        data = request.json
        item_name = data.get('item_name')
        dispatch_location = data.get('dispatch_location')
        delivery_location = data.get('delivery_location')
        
        if not all([item_name, dispatch_location, delivery_location]):
            return jsonify({'error': 'All fields are required'}), 400
        
        input_features = pd.DataFrame({
            'Item Name': [item_name],
            'Dispatch Location': [dispatch_location],
            'Delivery Location': [delivery_location]
        })
        
        for column in input_features.columns:
            if column in label_encoders:
                classes = list(label_encoders[column].classes_)
                if input_features[column].iloc[0] not in classes:
                    input_features[column] = [classes[0]]
                input_features[column] = label_encoders[column].transform(input_features[column])
        
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0]
        prediction_label = label_encoders['target'].inverse_transform([prediction])[0]
        confidence = probability[prediction] * 100
        
        resell_recommendation = "Yes" if probability[1] < 0.3 else "No"
        resell_confidence = (1 - probability[1]) * 100
        
        risk_level = "High" if probability[1] > 0.7 else "Medium" if probability[1] > 0.3 else "Low"
        
        return jsonify({
            'prediction': prediction_label,
            'confidence': round(confidence, 2),
            'return_probability': round(probability[1] * 100, 2),
            'resell_recommendation': resell_recommendation,
            'resell_confidence': round(resell_confidence, 2),
            'risk_level': risk_level
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
