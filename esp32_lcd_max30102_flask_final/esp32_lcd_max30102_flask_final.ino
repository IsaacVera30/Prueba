
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include "MAX30105.h"
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>

#define TFT_CS     15
#define TFT_DC     2
#define TFT_RST    4
#define TFT_BL     32

Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);

MAX30105 sensor;
const char* ssid = "Extence";
const char* password = "@Isaac20";
const char* server = "http://192.168.0.102:5000/api/datos";
const char* auth_url = "http://192.168.0.102:5000/api/autorizacion";

unsigned long lastBeat = 0;
float beatsPerMinute;
int beatAvg = 0;
int spo2 = 0;

#define MUESTRAS 10
int irMuestras[MUESTRAS];
int redMuestras[MUESTRAS];
int index = 0;

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi conectado");

  if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("❌ Sensor MAX30102 no detectado.");
    while (1);
  }

  sensor.setup();
  sensor.setPulseAmplitudeRed(0x3F);
  sensor.setPulseAmplitudeGreen(0);

  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH);

  tft.init(135, 240);
  tft.setRotation(3);
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(2);
}

int maxArray(int *arr) {
  int maxVal = arr[0];
  for (int i = 1; i < MUESTRAS; i++)
    if (arr[i] > maxVal) maxVal = arr[i];
  return maxVal;
}

int minArray(int *arr) {
  int minVal = arr[0];
  for (int i = 1; i < MUESTRAS; i++)
    if (arr[i] < minVal) minVal = arr[i];
  return minVal;
}

float promedio(int *arr) {
  long suma = 0;
  for (int i = 0; i < MUESTRAS; i++) suma += arr[i];
  return suma / (float)MUESTRAS;
}

bool autorizadoFlask() {
  HTTPClient http;
  http.begin(auth_url);
  int code = http.GET();
  bool autorizado = false;
  if (code == 200) {
    String payload = http.getString();
    autorizado = payload.indexOf("true") >= 0;
  }
  http.end();
  return autorizado;
}

void mostrarLCD(int hr, int sp, int ir, int red) {
  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(0, 0);
  tft.print("❤️ HR: ");
  tft.println(hr);
  tft.print("🩸 SpO2: ");
  tft.print(sp);
  tft.println("%");
  tft.print("IR avg: ");
  tft.println(ir);
  tft.print("RED avg: ");
  tft.println(red);
}

void loop() {
  long ir = sensor.getIR();
  long red = sensor.getRed();

  irMuestras[index] = ir;
  redMuestras[index] = red;
  index = (index + 1) % MUESTRAS;

  if (ir > 10000 && (millis() - lastBeat) > 300) {
    unsigned long delta = millis() - lastBeat;
    lastBeat = millis();
    beatsPerMinute = 60.0 / (delta / 1000.0);
    if (beatsPerMinute > 30 && beatsPerMinute < 180) {
      beatAvg = (int)beatsPerMinute;
    }
  }

  if (index == 0) {
    float avgIR = promedio(irMuestras);
    float avgRED = promedio(redMuestras);
    float acIR = maxArray(irMuestras) - minArray(irMuestras);
    float acRED = maxArray(redMuestras) - minArray(redMuestras);

    float ratio = (acRED / avgRED) / (acIR / avgIR);
    spo2 = constrain(110 - 25 * ratio, 70, 100);

    mostrarLCD(beatAvg, spo2, (int)avgIR, (int)avgRED);

    if (autorizadoFlask()) {
      HTTPClient http;
      http.begin(server);
      http.addHeader("Content-Type", "application/json");
      String payload = "{"hr":" + String(beatAvg) +
                       ","spo2":" + String(spo2) +
                       ","ir":" + String((int)avgIR) +
                       ","red":" + String((int)avgRED) + "}";
      int code = http.POST(payload);
      if (code > 0) {
        Serial.println("📡 Respuesta servidor:");
        Serial.println(http.getString());
      } else {
        Serial.print("❌ HTTP error: ");
        Serial.println(code);
      }
      http.end();
    } else {
      Serial.println("⚠️ No autorizado.");
    }
  }

  delay(10);
}
