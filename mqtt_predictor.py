import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import paho.mqtt.client as mqtt
import time
from datetime import datetime, timedelta

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model class
class TemperatureNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.layer3 = nn.Linear(32, 16)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(16, 1)
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.act3(self.layer3(x))
        return self.output(x)

# Load model
checkpoint = torch.load('algeria_hourly_temperature_model.pth')
input_size = checkpoint['input_size']
model = TemperatureNN(input_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(device)

# Load scaler
with open('hourly_temperature_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

feature_columns = checkpoint['feature_columns']
print(f"Loaded model with features: {feature_columns}")

# MQTT setup
broker = "06bc4541ea0049c68e8fdeb06d156411.s1.eu.hivemq.cloud"
port = 8883
topic = "algiers/temperature"
user = "ai_model"
password = "Password1"

# MQTT client
client = mqtt.Client(client_id="temperature_predictor_" + str(time.time()))
client.username_pw_set(user, password)
client.tls_set()
client.on_connect = lambda c, u, f, rc: print(f"Connected with code {rc}" if rc == 0 else f"Connection failed: {rc}")
client.on_publish = lambda c, u, mid: print(f"Published message ID {mid}")

# Connect
try:
    client.connect(broker, port)
    client.loop_start()
except Exception as e:
    print(f"Connection error: {e}")
    exit(1)

# Prediction function
def predict_temperature(model, scaler, date_str, hour, feature_columns):
    date = pd.to_datetime(date_str)
    input_features = {
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'dayofyear': date.dayofyear,
        'hour': hour,
        'sin_hour': np.sin(2 * np.pi * hour / 24),
        'cos_hour': np.cos(2 * np.pi * hour / 24),
        'season_0': 1 if date.month in [12, 1, 2] else 0,
        'season_1': 1 if date.month in [3, 4, 5] else 0,
        'season_2': 1 if date.month in [6, 7, 8] else 0,
        'season_3': 1 if date.month in [9, 10, 11] else 0
    }
    input_df = pd.DataFrame([input_features])[feature_columns]
    input_scaled = scaler.transform(input_df)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction

# Function to send prediction
def send_prediction(predict_time):
    date_str = predict_time.strftime('%Y-%m-%d')
    hour = predict_time.hour
    temp = predict_temperature(model, scaler, date_str, hour, feature_columns)
    datetime_str = predict_time.strftime('%Y-%m-%d %H:%M:%S')
    message = f'{{"temp": {temp:.2f}, "datetime": "{datetime_str}"}}'
    result = client.publish(topic, message, qos=1)
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"Published to {topic}: {datetime_str} - {temp:.2f}°C")
    else:
        print(f"Failed to publish: {result.rc}")

# Continuous predictions
try:
    predict_time = datetime.now() + timedelta(hours=1)  # Start 1 hour ahead
    predict_time = predict_time.replace(minute=0, second=0, microsecond=0)  # Align to hour
    while True:
        send_prediction(predict_time)
        predict_time += timedelta(hours=1)  # Move to next hour
        time.sleep(1)  # Wait 1 second between predictions
except KeyboardInterrupt:
    print("Stopping...")
finally:
    client.loop_stop()
    client.disconnect()

#tntafc123