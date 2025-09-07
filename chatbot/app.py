from flask import Flask, render_template, request, jsonify
import os
import sys
import math
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Add the current directory to the path to import from chatbot.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from chatbot.py
from chatbot import (
    load_text_knowledge,
    answer_query,
    safety_check,
    get_coordinates_from_pincode,
    check_earthquakes,
    get_weather
)

# ---------------- FLASK APP SETUP ----------------
app = Flask(__name__, template_folder='../templates')

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load knowledge base once at startup
try:
    knowledge = load_text_knowledge("knowledge.txt")
except FileNotFoundError:
    knowledge = "No knowledge base file found. Please ensure knowledge.txt exists in the chatbot directory."

# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictions')
def predictions():
    return render_template('prediction.html')

@app.route('/alerts')
def alerts():
    return render_template('alerts.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Handle chat messages from the frontend"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Check if it's a safety check request
        if "safe to work" in user_message.lower() or "safety check" in user_message.lower():
            # Try to extract pincode from the message
            words = user_message.split()
            pincode = None
            
            # Look for a 6-digit number (Indian pincode)
            for word in words:
                if word.isdigit() and len(word) == 6:
                    pincode = word
                    break
            
            if pincode:
                result = safety_check(pincode)
                return jsonify({
                    'response': result,
                    'type': 'safety_check'
                })
            else:
                return jsonify({
                    'response': "To check if it's safe to work, please provide your 6-digit PIN code. For example: 'Is it safe to work today? My pincode is 110001'",
                    'type': 'safety_prompt'
                })
        
        # Check if it's a weather request
        elif "weather" in user_message.lower():
            # Try to extract pincode from the message
            words = user_message.split()
            pincode = None
            
            for word in words:
                if word.isdigit() and len(word) == 6:
                    pincode = word
                    break
            
            if pincode:
                try:
                    user_lat, user_lon = get_coordinates_from_pincode(pincode)
                    temp_c, rain_mm = get_weather(user_lat, user_lon)
                    
                    weather_response = f"🌡️ Current Weather for PIN {pincode}:\n"
                    weather_response += f"Temperature: {temp_c}°C\n"
                    weather_response += f"Rainfall: {rain_mm} mm\n"
                    
                    if temp_c <= 0:
                        weather_response += "⚠️ Warning: Temperature is freezing"
                    elif temp_c >= 40:
                        weather_response += "⚠️ Warning: Temperature is very high"
                    
                    if rain_mm > 0:
                        weather_response += f"\n🌧️ Rain detected: {rain_mm} mm"
                    
                    return jsonify({
                        'response': weather_response,
                        'type': 'weather'
                    })
                except Exception as e:
                    return jsonify({
                        'response': f"❌ Error getting weather data: {str(e)}",
                        'type': 'error'
                    })
            else:
                return jsonify({
                    'response': "To get weather information, please provide your 6-digit PIN code. For example: 'What's the weather like in 110001?'",
                    'type': 'weather_prompt'
                })
        
        # For general knowledge queries, use the RAG system
        else:
            try:
                response = answer_query(user_message, knowledge)
                return jsonify({
                    'response': response,
                    'type': 'general'
                })
            except Exception as e:
                return jsonify({
                    'response': f"I'm sorry, I encountered an error while processing your request: {str(e)}",
                    'type': 'error'
                })
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    """Get suggested prompts for the user"""
    suggestions = [
        "Is it safe to work today? My pincode is 110001",
        "What is the current weather in 400001?",
        "What safety measures should I take during mining operations?",
        "How do I handle rockfall alerts?",
        "What are the common causes of mining accidents?",
        "Tell me about earthquake safety in mining",
        "What equipment is needed for safe mining?",
        "How often should safety inspections be conducted?",
        "What are the signs of unstable ground?",
        "How do I report a safety concern?"
    ]
    
    import random
    return jsonify({
        'suggestions': random.sample(suggestions, 4)
    })

# ---------------- ERROR HANDLERS ----------------

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ---------------- MAIN ----------------

if __name__ == '__main__':
    # Check if knowledge file exists
    if not os.path.exists('knowledge.txt'):
        print("Warning: knowledge.txt not found. Creating a sample knowledge base...")
        sample_knowledge = """
Mining Safety Guidelines:

1. Always wear proper safety equipment including helmets, boots, and protective clothing.
2. Conduct regular safety inspections of mining equipment and work areas.
3. Monitor weather conditions as they can affect mining operations.
4. Be aware of geological hazards including rockfall, landslides, and unstable ground.
5. Maintain proper ventilation in underground mining operations.
6. Follow emergency procedures and evacuation plans.
7. Report any safety concerns immediately to supervisors.
8. Never work alone in potentially hazardous areas.
9. Keep emergency communication devices accessible at all times.
10. Regular training on safety protocols is essential for all mining personnel.

Rockfall Prevention:
- Monitor rock face stability regularly
- Install protective barriers where necessary
- Avoid working during adverse weather conditions
- Use proper scaling techniques to remove loose rocks
- Implement early warning systems for unstable areas

Weather Considerations:
- Heavy rainfall can increase rockfall risk
- Extreme temperatures can affect equipment performance
- Wind conditions may impact crane and lifting operations
- Lightning poses electrical hazards to mining equipment
        """
        
        with open('knowledge.txt', 'w', encoding='utf-8') as f:
            f.write(sample_knowledge)
        
        print("Sample knowledge.txt created. You can edit this file to add more specific information.")
    
    print("Starting MineSafe Chatbot Server...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
