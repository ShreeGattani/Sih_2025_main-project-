# MineSafe Chatbot Integration

This project integrates the chatbot.py backend with the chatbot.html frontend using Flask.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   cd chatbot
   pip install -r requirements.txt
   ```

2. **Environment Setup**
   - Copy `env.template` to `.env`
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_actual_openai_api_key_here
     ```

3. **Knowledge Base**
   - The app will create a sample `knowledge.txt` file if it doesn't exist
   - You can edit this file to add mining-specific knowledge

4. **Run the Application**
   ```bash
   python app.py
   ```
   
   The application will be available at `http://localhost:5000`

## Features

### 🤖 **Intelligent Chatbot**
- **RAG (Retrieval-Augmented Generation)**: Uses knowledge.txt for mining-specific responses
- **Safety Checks**: Real-time safety assessment based on location and weather
- **Weather Integration**: Current weather conditions for any Indian PIN code
- **Earthquake Monitoring**: Checks for recent earthquakes within 50km radius

### 🛡️ **Safety Features**
- **PIN Code Safety Check**: `"Is it safe to work today? My pincode is 110001"`
- **Weather Queries**: `"What's the weather like in 400001?"`
- **Risk Assessment**: Combines weather, earthquake, and environmental data

### 💬 **Chat Interface**
- **Real-time messaging** with typing indicators
- **Smart suggestions** for common mining safety queries
- **Formatted responses** for safety checks and weather data
- **Quick prompts** for common questions

## API Endpoints

- `GET /chatbot` - Main chatbot interface
- `POST /api/chat` - Send messages to the chatbot
- `GET /api/suggestions` - Get random suggestions

## Message Types

1. **Safety Checks**: Include "safe to work" + 6-digit PIN code
2. **Weather Queries**: Include "weather" + 6-digit PIN code  
3. **General Questions**: Any mining safety related questions

## Example Queries

```
"Is it safe to work today? My pincode is 110001"
"What's the weather like in 400001?"
"What safety equipment is required for underground mining?"
"How do I handle a rockfall alert?"
"What are the signs of unstable ground?"
```

## File Structure

```
chatbot/
├── app.py                 # Flask web application
├── chatbot.py            # Original chatbot logic
├── requirements.txt      # Python dependencies
├── env.template         # Environment variables template
├── knowledge.txt        # Knowledge base (auto-generated)
└── README.md           # This file

templates/
└── chatbot.html         # Frontend interface
```

## Troubleshooting

1. **Import Errors**: Make sure all dependencies are installed
2. **API Key Issues**: Verify your OpenAI API key in the .env file
3. **Port Issues**: Change the port in app.py if 5000 is already in use
4. **Knowledge Base**: Ensure knowledge.txt exists or let the app create it

## Customization

- **Add Knowledge**: Edit `knowledge.txt` to include more mining-specific information
- **Modify Prompts**: Update quick prompts in `chatbot.html`
- **Styling**: Customize the CSS in `chatbot.html`
- **API Integration**: Add more APIs in `app.py` for additional data sources
