#include <dummy.h>

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <DHT.h>
#include <SPI.h>
#include <MFRC522.h>
#include <EEPROM.h>



// Definitions
#define DHTPIN 4
#define DHTTYPE DHT11
#define MQ2PIN 36
#define FLAME_PIN 2
#define PIR1_PIN 19
#define BUZZER_DOOR 21
#define BUZZER_MAINT 7    // Updated from 27
#define FAN_PIN 15

// RC522 Pins (Software SPI)
#define RC522_SS 5
#define RC522_RST 10
#define RC522_SCK 18
#define RC522_MISO 40
#define RC522_MOSI 20

// I2C Pins for LCD
#define SDA_PIN 8
#define SCL_PIN 9


// Thresholds

// Objects
DHT dht(DHTPIN, DHTTYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2);
MFRC522 mfrc522(RC522_SS, RC522_RST);

// Thresholds
float maxTemp = 30.0;
float maxHum = 70.0;
int maxGas = 300;

// EEPROM addresses
int addrTemp = 0;
int addrHum = sizeof(float);
int addrGas = sizeof(float) * 2;

void setup() {
  Serial.begin(115200);
  dht.begin();
  lcd.init();
  lcd.backlight();
  Wire.begin(SDA_PIN, SCL_PIN); // Explicit I2C initialization
  SPI.begin(RC522_SCK, RC522_MISO, RC522_MOSI, RC522_SS); // Software SPI

  mfrc522.PCD_Init();

  pinMode(FLAME_PIN, INPUT);
  pinMode(PIR1_PIN, INPUT);
  pinMode(MQ2PIN, INPUT);
  pinMode(BUZZER_DOOR, OUTPUT);
  pinMode(BUZZER_MAINT, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);

  digitalWrite(BUZZER_DOOR, LOW);
  digitalWrite(BUZZER_MAINT, LOW);
  digitalWrite(FAN_PIN, LOW);

  loadThresholds();

  lcd.setCursor(0, 0);
  lcd.print("System Started");
  delay(2000);
  lcd.clear();
}

void loop() {
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  int gasValue = analogRead(MQ2PIN);
  bool flame = digitalRead(FLAME_PIN);
  bool motion1 = digitalRead(PIR1_PIN);

  lcd.setCursor(0, 0);
  lcd.print("T:");
  lcd.print(temp);
  lcd.print(" H:");
  lcd.print(hum);

  lcd.setCursor(0, 1);
  lcd.print("G:");
  lcd.print(gasValue);
  lcd.print(" F:");
  lcd.print(flame ? "YES" : "NO");

  if (temp > maxTemp || hum > maxHum || gasValue > maxGas || flame) {
    digitalWrite(BUZZER_MAINT, HIGH);
  } else {
    digitalWrite(BUZZER_MAINT, LOW);
  }

  if (temp > maxTemp) {
    digitalWrite(FAN_PIN, HIGH);
  } else {
    digitalWrite(FAN_PIN, LOW);
  }

  if (motion1) {
    digitalWrite(BUZZER_DOOR, HIGH);
  } else {
    digitalWrite(BUZZER_DOOR, LOW);
  }

  checkRFID();

  delay(1000);
}

void checkRFID() {
  if (!mfrc522.PICC_IsNewCardPresent()) return;
  if (!mfrc522.PICC_ReadCardSerial()) return;

  Serial.print("Card UID: ");
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    Serial.print(mfrc522.uid.uidByte[i], HEX);
  }
  Serial.println();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("RFID Detected!");
  delay(1500);
  lcd.clear();
}

void saveThresholds() {
  EEPROM.put(addrTemp, maxTemp);
  EEPROM.put(addrHum, maxHum);
  EEPROM.put(addrGas, maxGas);
}

void loadThresholds() {
  EEPROM.get(addrTemp, maxTemp);
  EEPROM.get(addrHum, maxHum);
  EEPROM.get(addrGas, maxGas);
}

void setThresholds() {
  Serial.println("Enter max temperature (°C): ");
  while (Serial.available() == 0) {}
  maxTemp = Serial.readStringUntil('\n').toFloat();

  Serial.println("Enter max humidity (%): ");
  while (Serial.available() == 0) {}
  maxHum = Serial.readStringUntil('\n').toFloat();

  Serial.println("Enter max gas value (0-1023): ");
  while (Serial.available() == 0) {}
  maxGas = Serial.readStringUntil('\n').toInt();

  saveThresholds();
  Serial.println("Thresholds updated successfully!");

  Serial.print("Max Temp: "); Serial.println(maxTemp);
  Serial.print("Max Hum: "); Serial.println(maxHum);
  Serial.print("Max Gas: "); Serial.println(maxGas);
}