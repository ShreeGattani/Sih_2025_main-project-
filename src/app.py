from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
from test_api import check_earthquakes, get_weather 

app = Flask(__name__, template_folder='../templates')

# -------- Alert Counter --------
class AlertCounter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1
    def get_count(self):
        return self.count

alert_counter = AlertCounter()

# -------- Load model and data --------
model_path = '../models/rock_fall_prediction_model.pkl'

try:
    # Load the new model structure
    model_data = joblib.load(model_path)
    best_model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    engineered_features = model_data['engineered_features']
    risk_thresholds = model_data['risk_thresholds']
    
    print("Model loaded successfully!")
    print(f"Model type: {type(best_model)}")
    print(f"Feature names: {feature_names}")
    print(f"Engineered features: {engineered_features}")
    print(f"Risk thresholds: {risk_thresholds}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback - you might want to handle this differently
    best_model = None
    scaler = None
    label_encoder = None
    feature_names = []
    engineered_features = []
    risk_thresholds = {}

# Load test data
try:
    test_data = pd.read_csv('../models/test1.csv', encoding='utf-8')
    print(f"Test data loaded: {test_data.shape}")
    print(f"Test data columns: {test_data.columns.tolist()}")
except UnicodeDecodeError:
    try:
        test_data = pd.read_csv('../models/test1.csv', encoding='latin-1')
        print(f"Test data loaded with latin-1 encoding: {test_data.shape}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        # Create dummy test data if file doesn't exist
        test_data = pd.DataFrame({
            'slope_angle': [45.0, 50.0, 35.0],
            'slope_height': [100.0, 120.0, 80.0],
            'pore_pressure_kPa': [50.0, 60.0, 40.0],
            'strain_micro': [100.0, 120.0, 90.0],
            'rainfall_mm': [10.0, 15.0, 5.0],
            'temperature_Cvibration_mmps': [25.0, 30.0, 20.0],
            'joint_density': [5.0, 7.0, 3.0],
            'crack_length': [2.0, 3.0, 1.0],
            'displacement_mm': [5.0, 8.0, 3.0]
        })
        print("Using dummy test data")

def create_engineered_features(data):
    """Create the same engineered features as in training"""
    data_copy = data.copy()
    
    # Stability ratio
    if 'slope_angle' in data_copy.columns and 'slope_height' in data_copy.columns:
        data_copy['stability_ratio'] = data_copy['slope_height'] / np.tan(
            np.radians(data_copy['slope_angle'].clip(1, 89))  # Avoid division by zero
        )
    
    # Effective stress
    if 'pore_pressure_kPa' in data_copy.columns and 'strain_micro' in data_copy.columns:
        data_copy['effective_stress'] = data_copy['strain_micro'] - (data_copy['pore_pressure_kPa'] / 1000)
    
    # Weather stress
    if 'rainfall_mm' in data_copy.columns and 'temperature_Cvibration_mmps' in data_copy.columns:
        data_copy['weather_stress'] = (data_copy['rainfall_mm'] * 0.7 + 
                                      data_copy['temperature_Cvibration_mmps'] * 0.3)
    
    # Rock quality
    if 'joint_density' in data_copy.columns and 'crack_length' in data_copy.columns:
        data_copy['rock_quality'] = 100 / (1 + data_copy['joint_density'] + data_copy['crack_length'] / 10)
    
    # Displacement severity
    if 'displacement_mm' in data_copy.columns:
        data_copy['displacement_severity'] = pd.cut(data_copy['displacement_mm'],
                                                   bins=[0, 5, 15, float('inf')],
                                                   labels=[1, 2, 3])
        data_copy['displacement_severity'] = data_copy['displacement_severity'].astype(float)
    
    return data_copy

def prepare_prediction_data(data):
    """Prepare data for prediction using the same process as training"""
    if best_model is None:
        return None
    
    # Create engineered features
    processed_data = create_engineered_features(data)
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in processed_data.columns:
            print(f"Missing feature '{feature}', filling with median/mode")
            # Fill missing features with reasonable defaults
            if feature in ['stability_ratio', 'effective_stress', 'weather_stress', 'rock_quality']:
                processed_data[feature] = 0.0
            elif feature == 'displacement_severity':
                processed_data[feature] = 1.0
            else:
                processed_data[feature] = 0.0
    
    # Select only the features the model expects
    processed_data = processed_data[feature_names]
    
    # Handle any remaining missing values
    processed_data = processed_data.fillna(0)
    
    # Apply scaling if needed (for SVM, Neural Network models)
    # The model wrapper should handle this automatically
    
    return processed_data

# -------- Prepare predictions --------
if best_model is not None:
    try:
        # Prepare test data for prediction
        X_test_prepared = prepare_prediction_data(test_data)
        
        if X_test_prepared is not None:
            # Use only first 5 rows for initial predictions
            sample_size = min(5, len(X_test_prepared))
            sample_data = X_test_prepared.iloc[:sample_size]
            
            # Make predictions
            predictions = best_model.predict(sample_data)
            probs = best_model.predict_proba(sample_data)
            
            # Convert predictions back to original labels
            if hasattr(label_encoder, 'inverse_transform'):
                prediction_labels = label_encoder.inverse_transform(predictions)
            else:
                prediction_labels = predictions
            
            # Get class names
            if hasattr(label_encoder, 'classes_'):
                class_names = label_encoder.classes_
            else:
                class_names = ["Low", "Medium", "High", "Critical"]
            
            # Create probability dictionaries
            prob_dicts = []
            probs_list = []
            
            for i, sample_probs in enumerate(probs):
                prob_dict = {}
                max_prob = 0
                for j, class_name in enumerate(class_names):
                    prob_value = float(sample_probs[j])
                    prob_dict[class_name] = round(prob_value, 4)
                    max_prob = max(max_prob, prob_value)
                
                prob_dicts.append(prob_dict)
                probs_list.append(max_prob)
            
            print("Predictions successful!")
            print(f"Sample predictions: {prediction_labels}")
            print(f"Sample probabilities: {prob_dicts[0] if prob_dicts else 'None'}")
            
        else:
            raise Exception("Failed to prepare prediction data")
            
    except Exception as e:
        print(f"Prediction error: {e}")
        # Fallback to dummy data
        probs_list = [0.5]
        prob_dicts = [{"Low": 0.25, "Medium": 0.25, "High": 0.25, "Critical": 0.25}]
        prediction_labels = ["Medium"]
else:
    print("Model not available, using dummy predictions")
    probs_list = [0.5]
    prob_dicts = [{"Low": 0.25, "Medium": 0.25, "High": 0.25, "Critical": 0.25}]
    prediction_labels = ["Medium"]

# -------- Risk determination --------
RISK_LEVELS = {
    "Low": 0.25,
    "Medium": 0.5,
    "High": 0.75,
    "Critical": 1.0,
}

def determine_risk(prob: float) -> str:
    """Return risk category based on probability (0–1)"""
    if prob < 0.25:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"

def predict_rock_fall_risk(input_data):
    """Predict rock fall risk for new data"""
    if best_model is None:
        return "Unknown", {"Low": 0.25, "Medium": 0.25, "High": 0.25, "Critical": 0.25}
    
    try:
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data
        
        # Prepare data for prediction
        prepared_data = prepare_prediction_data(input_df)
        
        if prepared_data is None:
            raise Exception("Failed to prepare input data")
        
        # Make prediction
        prediction = best_model.predict(prepared_data)
        probabilities = best_model.predict_proba(prepared_data)
        
        # Convert to original labels
        if hasattr(label_encoder, 'inverse_transform'):
            risk_level = label_encoder.inverse_transform(prediction)[0]
        else:
            risk_level = prediction[0]
        
        # Create probability dictionary
        prob_dict = {}
        class_names = label_encoder.classes_ if hasattr(label_encoder, 'classes_') else ["Low", "Medium", "High", "Critical"]
        
        for i, class_name in enumerate(class_names):
            prob_dict[class_name] = float(probabilities[0][i])
        
        return risk_level, prob_dict
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown", {"Low": 0.25, "Medium": 0.25, "High": 0.25, "Critical": 0.25}

# -------- Routes --------
@app.route("/")
@app.route("/index")
def index():
    if probs_list:
        risk = determine_risk(probs_list[0])
    else:
        risk = "Unknown"
    return render_template("index.html", risk=risk, no_of_alerts=alert_counter.get_count())

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for rock fall prediction"""
    try:
        data = request.get_json()
        risk_level, probabilities = predict_rock_fall_risk(data)
        
        # Check if this is a high risk situation
        if risk_level in ["High", "Critical"] or probabilities.get("High", 0) > 0.6 or probabilities.get("Critical", 0) > 0.3:
            alert_counter.increment()
        
        return jsonify({
            "status": "success",
            "risk_level": risk_level,
            "probabilities": probabilities,
            "alert_count": alert_counter.get_count()
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# New route to receive location
@app.route("/location", methods=["POST"])
def receive_location():
    data = request.get_json()
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    print(f"Received location: {latitude}, {longitude}")
    response = {"status": "success", "reasons": [], "safe": True}
    try:
        # Earthquake check
        if check_earthquakes(float(latitude), float(longitude)):
            response["reasons"].append("Earthquake detected within 50 km")
            response["safe"] = False
            alert_counter.increment()
        
        # Weather check
        temp_c, rain_mm = get_weather(float(latitude), float(longitude))
        if temp_c <= 0:
            response["reasons"].append(f"Temperature too low: {temp_c}°C")
            response["safe"] = False
        elif temp_c >= 40:
            response["reasons"].append(f"Temperature too high: {temp_c}°C")
            response["safe"] = False
        if rain_mm > 0:
            response["reasons"].append(f"Rainfall: {rain_mm} mm")
            response["safe"] = False
        
        response["temperature"] = temp_c
        response["rainfall"] = rain_mm
        
        # Add rock fall risk assessment based on weather
        weather_input = {
            "rainfall_mm": rain_mm,
            "temperature_Cvibration_mmps": temp_c,
            "slope_angle": 45.0,  # Default values - in real app, get from user/database
            "slope_height": 100.0,
            "pore_pressure_kPa": 50.0,
            "strain_micro": 100.0,
            "joint_density": 5.0,
            "crack_length": 2.0,
            "displacement_mm": 5.0
        }
        
        risk_level, probabilities = predict_rock_fall_risk(weather_input)
        response["rock_fall_risk"] = risk_level
        response["rock_fall_probabilities"] = probabilities
        
        if risk_level in ["High", "Critical"]:
            response["reasons"].append(f"High rock fall risk detected: {risk_level}")
            response["safe"] = False
            
    except Exception as e:
        response["status"] = "error"
        response["error"] = str(e)
        response["safe"] = False
    
    print(response)
    return jsonify(response)

@app.route("/alerts", methods=["GET", "POST"])
def alerts():
    alerts_list = []
    latitude = None
    longitude = None
    
    if request.method == "POST":
        data = request.form
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        
        if latitude and longitude:
            try:
                import datetime
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Earthquake alert
                if check_earthquakes(float(latitude), float(longitude)):
                    alerts_list.append({
                        "type": "Earthquake",
                        "message": "Earthquake detected within 50 km of your location.",
                        "timestamp": now
                    })
                    alert_counter.increment()
                
                # Weather alert
                temp_c, rain_mm = get_weather(float(latitude), float(longitude))
                if rain_mm > 20:  # Heavy rainfall threshold
                    alerts_list.append({
                        "type": "Heavy Rainfall",
                        "message": f"Heavy rainfall detected: {rain_mm} mm.",
                        "timestamp": now
                    })
                    alert_counter.increment()
                
                # Rock fall risk alert
                weather_input = {
                    "rainfall_mm": rain_mm,
                    "temperature_Cvibration_mmps": temp_c,
                    "slope_angle": 45.0,
                    "slope_height": 100.0,
                    "pore_pressure_kPa": 50.0,
                    "strain_micro": 100.0,
                    "joint_density": 5.0,
                    "crack_length": 2.0,
                    "displacement_mm": 5.0
                }
                
                risk_level, probabilities = predict_rock_fall_risk(weather_input)
                
                if risk_level == "Critical":
                    alerts_list.append({
                        "type": "Critical Rock Fall Risk",
                        "message": f"CRITICAL rock fall risk detected! Probability: {probabilities.get('Critical', 0):.2%}",
                        "timestamp": now
                    })
                    alert_counter.increment()
                elif risk_level == "High":
                    alerts_list.append({
                        "type": "High Rock Fall Risk",
                        "message": f"High rock fall risk detected! Probability: {probabilities.get('High', 0):.2%}",
                        "timestamp": now
                    })
                    alert_counter.increment()
                
            except Exception as e:
                alerts_list.append({
                    "type": "Error",
                    "message": str(e),
                    "timestamp": now
                })
                alert_counter.increment()
    
    return render_template("alerts.html", alerts=alerts_list, latitude=latitude, 
                         longitude=longitude, alert_count=alert_counter.get_count())

@app.route("/predictions")
def predictions():
    return render_template("prediction.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/results")
def results():
    # Pass some sample results to the template
    sample_results = {
        "predictions": prediction_labels[:3] if 'prediction_labels' in globals() else ["Medium", "Low", "High"],
        "probabilities": prob_dicts[:3] if prob_dicts else [{"Low": 0.25, "Medium": 0.25, "High": 0.25, "Critical": 0.25}],
        "risk_thresholds": risk_thresholds
    }
    return render_template("results.html", results=sample_results)

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

if __name__ == "__main__":
    app.run(debug=True)