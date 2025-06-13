// --- tesis9porciento_modificado.ino ---
// --- Versión Final Actualizada ---

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>

#define ID_PACIENTE 1
#define TFT_CS    15
#define TFT_DC    2
#define TFT_RST   4
#define TFT_BL    32
#define BUZZER_PIN 25
#define BUTTON_PIN 34

#define WIFI_SSID "Extence"
#define WIFI_PASSWORD "@Isaac20"
#define SERVER_URL "https://prueba-nno9.onrender.com"

MAX30105 particleSensor;
Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);

float lastPas = -1;
float lastPad = -1;
float lastSpo2 = -1;
float lastHr = -1;
float lastHrSensor = -1;
float lastSpo2Sensor = -1;
long lastIR = 0;
long lastRED = 0;
String lastNivel = "---";
bool autorizado = false;
unsigned long ultimoEnvioAlServidor = 0;
const unsigned long intervaloEnvio = 200;

#define BUFFER_SIZE 100
uint32_t irBuffer[BUFFER_SIZE];
uint32_t redBuffer[BUFFER_SIZE];
int32_t spo2;
int8_t validSPO2;
int32_t heartRate;
int8_t validHeartRate;

const int NUM_LECTURAS = 5;
int lecturasHR[NUM_LECTURAS];
int indiceLectura = 0;
int totalHR = 0;
int promedioHR = 0;
int lecturasValidas = 0;

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH);
  tft.init(135, 240);
  tft.setRotation(1);
  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(10,10); tft.setTextSize(2); tft.setTextColor(ST77XX_WHITE);
  tft.println("Iniciando...");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  tft.setCursor(10,40); tft.setTextSize(1); tft.print("Conectando WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); tft.print(".");
  }
  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(10,30); tft.setTextColor(ST77XX_GREEN); tft.print("WiFi OK!");
  delay(1000);

  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    tft.fillScreen(ST77XX_BLACK);
    tft.setCursor(10,10); tft.setTextColor(ST77XX_RED); tft.setTextSize(2);
    tft.println("Error Sensor");
    while (1);
  }
  particleSensor.setup(80, 8, 2, 100, 118, 16384);
  for (int i = 0; i < NUM_LECTURAS; i++) lecturasHR[i] = 0;
}

void loop() {
  if (millis() - ultimoEnvioAlServidor > intervaloEnvio) {
    if (WiFi.status() == WL_CONNECTED) {
      for (int i = 0; i < BUFFER_SIZE; i++) {
        while (!particleSensor.available()) particleSensor.check();
        redBuffer[i] = particleSensor.getRed();
        irBuffer[i] = particleSensor.getIR();
        particleSensor.nextSample();
      }

      maxim_heart_rate_and_oxygen_saturation(irBuffer, BUFFER_SIZE, redBuffer, &spo2, &validSPO2, &heartRate, &validHeartRate);

      if (irBuffer[BUFFER_SIZE - 1] > 50000 && validHeartRate == 1 && heartRate > 40 && heartRate < 130 && validSPO2 == 1) {
        totalHR -= lecturasHR[indiceLectura];
        lecturasHR[indiceLectura] = heartRate;
        totalHR += lecturasHR[indiceLectura];
        indiceLectura = (indiceLectura + 1) % NUM_LECTURAS;
        if (lecturasValidas < NUM_LECTURAS) lecturasValidas++;
        promedioHR = totalHR / lecturasValidas;
        lastHrSensor = promedioHR;
        lastSpo2Sensor = spo2;
        lastIR = irBuffer[BUFFER_SIZE - 1];
        lastRED = redBuffer[BUFFER_SIZE - 1];
        enviarDatosAlServidor(promedioHR, lastIR, lastRED, spo2);
      }
    }
    ultimoEnvioAlServidor = millis();
  }

  while (particleSensor.available()) particleSensor.nextSample();
  mostrarDatosEnPantalla();
  delay(10);
}

void enviarDatosAlServidor(int hr_env, long ir_val, long red_val, float spo2_env) {
  HTTPClient http;
  String serverPath = String(SERVER_URL) + "/api/presion";
  http.begin(serverPath.c_str());
  http.addHeader("Content-Type", "application/json");

  // VERSIÓN COMPLETA Y CORRECTA DEL JSON QUE SE ENVÍA AL SERVIDOR
  String httpRequestData = "{";
  httpRequestData += "\"hr_crudo\":" + String(heartRate); // Valor de HR antes de promediar
  httpRequestData += ",\"hr_promedio\":" + String(hr_env);
  httpRequestData += ",\"spo2_sensor\":" + String(spo2_env);
  httpRequestData += ",\"ir\":" + String(ir_val);
  httpRequestData += ",\"red\":" + String(red_val);
  httpRequestData += ",\"id_paciente\":" + String(ID_PACIENTE); // ID para identificar el dispositivo/paciente
  httpRequestData += "}";

  int httpResponseCode = http.POST(httpRequestData);
  if (httpResponseCode == 200) {
    String payload = http.getString();
    int sysIdx = payload.indexOf("\"sys\":") + 6;
    int diaIdx = payload.indexOf("\"dia\":") + 6;
    int spo2Idx = payload.indexOf("\"spo2\":") + 7;
    int hrIdx = payload.indexOf("\"hr\":") + 5;
    int nivelIdx = payload.indexOf("\"nivel\":\"") + 9;

    if (sysIdx > 5) {
      lastPas = payload.substring(sysIdx, payload.indexOf(",", sysIdx)).toFloat();
      lastPad = payload.substring(diaIdx, payload.indexOf(",", diaIdx)).toFloat();
      lastSpo2 = payload.substring(spo2Idx, payload.indexOf(",", spo2Idx)).toFloat();
      lastHr = payload.substring(hrIdx, payload.indexOf(",", hrIdx)).toFloat();
      lastNivel = payload.substring(nivelIdx, payload.indexOf("\"", nivelIdx));
      digitalWrite(BUZZER_PIN, (lastNivel == "HT Crisis") ? HIGH : LOW);
    } else {
      lastNivel = "Error JSON";
      digitalWrite(BUZZER_PIN, LOW);
    }
  } else {
    lastNivel = "Error Com.";
    digitalWrite(BUZZER_PIN, LOW);
  }
  http.end();
}

void mostrarDatosEnPantalla() {
  tft.fillScreen(ST77XX_BLACK);

  // --- Datos del Sensor ---
  tft.setTextSize(1);
  tft.setCursor(5, 5); tft.setTextColor(ST77XX_GREEN);
  tft.println("**** Lectura del Sensor ****");
  tft.print("HR Crudo: "); tft.println(heartRate);
  tft.print("HR PROMEDIO SENSOR: "); tft.println(lastHrSensor);
  tft.print("SpO2 SENSOR: "); tft.print(lastSpo2Sensor); tft.println(" %");
  tft.print("IR: "); tft.println(lastIR);
  tft.print("RED: "); tft.println(lastRED);

  // --- Datos ML ---
  int yInicioML = 70;
  tft.setTextSize(1);
  tft.setCursor(5, yInicioML); tft.setTextColor(ST77XX_CYAN);
  tft.println("*** Estimacion ML ***");
  tft.setTextColor(ST77XX_YELLOW); tft.setCursor(5, yInicioML + 12);
  tft.print("SYS: "); tft.println((lastPas > 0) ? (int)lastPas : 0);
  tft.setCursor(90, yInicioML + 12); // Mostrar estado en misma fila que SYS
  if (lastNivel == "HT Crisis") tft.setTextColor(ST77XX_RED);
  else if (lastNivel == "HT2") tft.setTextColor(0xFDA0);
  else if (lastNivel == "HT1") tft.setTextColor(ST77XX_YELLOW);
  else if (lastNivel == "Elevada") tft.setTextColor(0x07E0);
  else if (lastNivel == "Normal") tft.setTextColor(ST77XX_GREEN);
  else tft.setTextColor(ST77XX_RED);
  tft.print("Estado: "); tft.println(lastNivel);

  tft.setTextColor(ST77XX_MAGENTA); tft.setCursor(5, yInicioML + 24);
  tft.print("DIA: "); tft.println((lastPad > 0) ? (int)lastPad : 0);
  tft.setTextColor(ST77XX_ORANGE); tft.setCursor(5, yInicioML + 36);
  tft.print("HR ML: "); tft.println((lastHr > 0) ? (int)lastHr : 0);
  tft.setTextColor(ST77XX_CYAN); tft.setCursor(5, yInicioML + 48);
  tft.print("SpO2 ML: "); tft.println((lastSpo2 > 0) ? (int)lastSpo2 : 0);
}
