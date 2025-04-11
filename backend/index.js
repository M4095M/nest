require('dotenv').config();
const express = require('express');
const { PrismaClient } = require('./generated/prisma-client');
const mqtt = require('mqtt');
const app = express();
const prisma = new PrismaClient();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// MQTT Setup for HiveMQ Cloud
const mqttClient = mqtt.connect({
  host: '41cc8a0bbb1a4924947dc1ea1afbef36.s1.eu.hivemq.cloud',
  port: 8883,
  protocol: 'mqtts', // Use MQTT over TLS
  username: process.env.MQTT_USERNAME, // Add to .env
  password: process.env.MQTT_PASSWORD, // Add to .env
});

mqttClient.on('connect', () => {
  console.log('Connected to MQTT broker');
  mqttClient.subscribe('algiers/temperature');
  mqttClient.subscribe('sensor/data'); // Keep existing topics if needed
  mqttClient.subscribe('ai/prediction');
});

mqttClient.on('message', async (topic, message) => {
  try {
    if (topic === 'algiers/temperature') {
      const temperature = parseFloat(message.toString());
      await prisma.temperaturePrediction.create({
        data: {
          temperature: temperature,
          predictedFor: new Date(), // Use current time as predictedFor (adjust as needed)
        },
      });
      console.log(`Stored temperature prediction: ${temperature}°C`);
    } else if (topic === 'sensor/data') {
      const data = JSON.parse(message.toString());
      await prisma.sensorData.create({
        data: {
          temperature: data.temperature,
          humidity: data.humidity,
        },
      });
      console.log(`Stored sensor data: Temp=${data.temperature}, Humidity=${data.humidity}`);
    } else if (topic === 'ai/prediction') {
      const data = JSON.parse(message.toString());
      await prisma.temperaturePrediction.create({
        data: {
          temperature: data.temperature,
          predictedFor: new Date(data.predictedFor),
        },
      });
      console.log(`Stored prediction: Temp=${data.temperature} for ${data.predictedFor}`);
    }
  } catch (err) {
    console.error('Error storing data:', err);
  }
});

// REST API Endpoints
app.get('/sensor-data', async (req, res) => {
  try {
    const sensorData = await prisma.sensorData.findMany();
    res.json(sensorData);
  } catch (err) {
    res.status(500).json({ message: 'Error retrieving sensor data' });
  }
});

app.get('/predictions', async (req, res) => {
  try {
    const predictions = await prisma.temperaturePrediction.findMany({
      orderBy: { createdAt: 'desc' },
    });
    res.json(predictions);
  } catch (err) {
    res.status(500).json({ message: 'Error retrieving predictions' });
  }
});

app.post('/publish', (req, res) => {
  const { topic, message } = req.body;
  if (!topic || !message) {
    return res.status(400).json({ message: 'Topic and message are required' });
  }
  mqttClient.publish(topic, JSON.stringify(message));
  res.send('Message published');
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});