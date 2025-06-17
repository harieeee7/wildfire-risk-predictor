# api/predict.py
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from joblib import load
import json

# Load your trained model once
model = load('wildfire_model.pkl')

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse query parameters: e.g. ?temperature=35&humidity=10
        query = parse_qs(urlparse(self.path).query)

        # Extract your input features safely
        try:
            temperature = float(query.get('temperature', [0])[0])
            humidity = float(query.get('humidity', [0])[0])
            precipitation = float(query.get('precipitation', [0])[0])
            wind_speed = float(query.get('wind_speed', [0])[0])
            landcover = float(query.get('landcover', [0])[0])
            elevation = float(query.get('elevation', [0])[0])
            ndvi = float(query.get('ndvi', [0])[0])
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid input: {str(e)}"}).encode())
            return

        # Prepare input as a list inside a list
        input_features = [[temperature, humidity, precipitation, wind_speed, landcover, elevation, ndvi]]
        prediction = model.predict(input_features)

        # Return JSON response
        self.send_response(200)
        self.send_header('Content-type','application/json')
        self.end_headers()
        response = json.dumps({
            "prediction": prediction.tolist()
        })
        self.wfile.write(response.encode())
        return
