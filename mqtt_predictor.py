import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import paho.mqtt.client as mqtt
import schedule
import time
from datetime import datetime

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model class (must match training)
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
broker = "41cc8a0bbb1a4924947dc1ea1afbef36.s1.eu.hivemq.cloud"  # Change to your MQTT broker address (e.g., "test.mosquitto.org")
port = 8883           # Default MQTT port
topic = "algiers/temperature"  # Topic to publish to
user = "comp_user"
password = "Password1"

# MQTT client
client = mqtt.Client()
client.connect(broker, port)
client.username_pw_set(user,password)


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
def send_prediction():
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    hour = now.hour
    temp = predict_temperature(model, scaler, date_str, hour, feature_columns)
    message = f"{temp:.2f}"
    client.publish(topic, message)
    print(f"Published to {topic}: {date_str} {hour:02d}:00 - {message}°C")

# Schedule to run every hour
schedule.every().hour.at(":00").do(send_prediction)

# Initial prediction
send_prediction()

# Keep script running
while True:
    schedule.run_pending()
    time.sleep(1)