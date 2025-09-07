import os
import math
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ---------------- LOAD ENV ----------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- CONFIG ----------------
EARTHQUAKE_API_KEY = "fb10922e-9fa9-4bfe-a44a-7aecc17d5234"
WEATHER_API_KEY = "620a81f2719043c0bfe180055250609"

# ---------------- RAG FUNCTIONS ----------------
def load_text_knowledge(file_path="knowledge.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def answer_query(query, knowledge):
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on the provided knowledge base."},
            {"role": "user", "content": f"Knowledge:\n{knowledge}\n\nQuestion: {query}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# ---------------- SAFETY FUNCTIONS ----------------
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in km"""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def get_coordinates_from_pincode(pincode):
    url = f"https://nominatim.openstreetmap.org/search?postalcode={pincode}&countrycodes=in&format=json"
    geo_resp = requests.get(url, headers={"User-Agent": "MyApp"}).json()
    if geo_resp:
        return float(geo_resp[0]["lat"]), float(geo_resp[0]["lon"])
    else:
        raise ValueError("Could not find coordinates for that PIN code")

def check_earthquakes(user_lat, user_lon, radius_km=50):
    url = "https://api.apiverve.com/v1/earthquake"
    headers = {"X-API-Key": EARTHQUAKE_API_KEY}
    resp = requests.get(url, headers=headers).json()

    if resp.get("status") != "ok":
        return False

    earthquakes = resp["data"].get("earthquakes", [])
    for quake in earthquakes:
        lon, lat = quake["coordinates"]
        distance = haversine(user_lat, user_lon, lat, lon)
        if distance <= radius_km:
            return True
    return False

def get_weather(user_lat, user_lon):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={user_lat},{user_lon}&aqi=no"
    resp = requests.get(url).json()

    if "error" in resp:
        msg = resp["error"].get("message", "Unknown error from WeatherAPI")
        raise ValueError(f"WeatherAPI error: {msg}")

    temp_c = resp["current"]["temp_c"]
    rain_mm = resp["current"].get("precip_mm", 0)
    return temp_c, rain_mm

def safety_check(pincode):
    try:
        user_lat, user_lon = get_coordinates_from_pincode(pincode)
    except ValueError as e:
        return f"❌ {e}"

    reasons = []

    # Earthquake check
    if check_earthquakes(user_lat, user_lon):
        reasons.append("- Earthquake detected within 50 km")

    # Weather check
    temp_c, rain_mm = get_weather(user_lat, user_lon)
    if temp_c <= 0:
        reasons.append(f"- Temperature too low: {temp_c}°C")
    elif temp_c >= 40:
        reasons.append(f"- Temperature too high: {temp_c}°C")
    if rain_mm > 0:
        reasons.append(f"- Rainfall: {rain_mm} mm")

    # Final response
    if reasons:
        return "\n❌ Not safe to work today.\nReasons:\n" + "\n".join(reasons)
    else:
        return f"\n✅ Safe to work today.\nNo rainfall.\nTemperature: {temp_c}°C (safe).\nNo earthquake within 50 km."

# ---------------- MAIN ----------------
def main():
    knowledge = load_text_knowledge("knowledge.txt")

    while True:
        user_input = input("\nAsk me something (or type 'exit'): ").strip()

        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break

        elif "safe to work" in user_input.lower():
            pincode = input("Enter your PIN code: ").strip()
            result = safety_check(pincode)
            print(result)

        else:
            result = answer_query(user_input, knowledge)
            print("🤖", result)

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()