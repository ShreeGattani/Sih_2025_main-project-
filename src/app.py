from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from test_api import check_earthquakes, get_weather 

app = Flask(__name__, template_folder='../templates')

# -------- Load model and data --------
model_path = '../models/rf_model.pkl'
with open(model_path, 'rb') as f:
    best_model = pickle.load(f)

# Load your single test file with proper encoding
try:
    test_data = pd.read_csv('../models/test_rock1.csv', encoding='utf-8')
except UnicodeDecodeError:
    test_data = pd.read_csv('../models/test_rock1.csv', encoding='latin-1')

# Get the feature names that the model expects
expected_features = best_model.feature_names_in_

print("Expected features:", expected_features)
print("Available features:", test_data.columns.tolist())
print("Expected feature count:", len(expected_features))
print("Available feature count:", len(test_data.columns))

# Create a mapping of available features to expected features
feature_mapping = {}
available_features = []

for expected in expected_features:
    found = False
    for available in test_data.columns:
        # Handle encoding issues with degree symbols
        if expected == available or \
           expected.replace('°', 'Â°') == available or \
           expected.replace('Â°', '°') == available:
            feature_mapping[expected] = available
            available_features.append(available)
            found = True
            break
    
    if not found:
        print(f"Missing feature: {expected}")

# Create X_test with available features and fill missing ones with zeros
X_test = pd.DataFrame(columns=expected_features)

# Fill in available features
for expected_feat, available_feat in feature_mapping.items():
    if available_feat in test_data.columns:
        X_test[expected_feat] = test_data[available_feat]

# Fill missing features with zeros or mean values
for feat in expected_features:
    if feat not in feature_mapping:
        print(f"Filling missing feature '{feat}' with zeros")
        X_test[feat] = 0.0

# Convert to numeric and handle any remaining issues
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

print("Final X_test shape:", X_test.shape)
print("Final X_test columns:", X_test.columns.tolist())

# For y_test, use the first column that's not used in X_test (likely 'Highwall')
unused_cols = [col for col in test_data.columns if col not in available_features]
if unused_cols:
    y_test = test_data[unused_cols[0]].values
    print(f"Using '{unused_cols[0]}' as target variable")
else:
    y_test = None

# -------- Prepare predictions --------
class_names = ["Low", "Medium", "High", "Critical"]

# Try to make predictions safely
try:
    # Use only first 5 rows for initial predictions
    sample_size = min(5, len(X_test))
    probs = best_model.predict_proba(X_test.iloc[:sample_size])
    classes = best_model.classes_
    
    # Flatten probabilities for first class (or choose the max probability per sample)
    probs_list = [max(sample) for sample in probs]
    prob_dicts = [
        {class_names[j]: round(float(sample[j]), 4) for j in range(len(classes))}
        for sample in probs
    ]
    print("Predictions successful!")
except Exception as e:
    print(f"Prediction error: {e}")
    # Fallback to dummy data
    probs_list = [0.5]
    prob_dicts = [{"Low": 0.25, "Medium": 0.25, "High": 0.25, "Critical": 0.25}]

# -------- Risk determination --------
RISK_LEVELS = {
    "Low": 0.25,
    "Medium": 0.5,
    "High": 0.75,
    "Critical": 1.0,
}

def determine_risk(prob: float) -> str:
    """Return risk category based on probability (0–1)"""
    for label, threshold in RISK_LEVELS.items():
        if prob < threshold:
            return label
    return "Critical"  # Changed from "Unknown" to "Critical" for safety

# -------- Routes --------
@app.route("/")
@app.route("/index")
def index():
    if probs_list:
        risk = determine_risk(probs_list[0])
    else:
        risk = "Unknown"
    return render_template("index.html", risk=risk)

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
                # Earthquake alert
                import datetime
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if check_earthquakes(float(latitude), float(longitude)):
                    alerts_list.append({
                        "type": "Earthquake",
                        "message": "Earthquake detected within 50 km of your location.",
                        "timestamp": now
                    })
                # Weather alert
                temp_c, rain_mm = get_weather(float(latitude), float(longitude))
                if rain_mm > 20:  # Heavy rainfall threshold (adjust as needed)
                    alerts_list.append({
                        "type": "Heavy Rainfall",
                        "message": f"Heavy rainfall detected: {rain_mm} mm.",
                        "timestamp": now
                    })
                # Thunderstorm alert (if you want to add, you need to check weather API for thunderstorm info)
                # For now, only earthquake and heavy rainfall
            except Exception as e:
                alerts_list.append({
                    "type": "Error",
                    "message": str(e),
                    "timestamp": now
                })
    return render_template("alerts.html", alerts=alerts_list, latitude=latitude, longitude=longitude)

@app.route("/predictions")
def predictions():
    return render_template("prediction.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

if __name__ == "__main__":
    app.run(debug=True)
