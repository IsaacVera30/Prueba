
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include "MAX30105.h"

#define LCD_CS   15
#define LCD_DC   2
#define LCD_RST  4
#define LCD_BLK  32
#define LCD_SCLK 18
#define LCD_MOSI 23

Adafruit_ST7789 lcd = Adafruit_ST7789(LCD_CS, LCD_DC, LCD_RST);
MAX30105 sensor;

const char* ssid = "Extence";
const char* password = "@Isaac20";
const char* server = "http://192.168.0.102:5000";

const int MUESTRAS = 10;
int irMuestras[MUESTRAS];
int redMuestras[MUESTRAS];
int index_muestra = 0;

unsigned long lastBeat = 0;
float beatsPerMinute = 0;
int beatAvg = 0;
int spo2 = 0;

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi conectado");

  if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("❌ MAX30102 no detectado");
    while (1);
  }

  sensor.setup();
  sensor.setPulseAmplitudeRed(0x3F);
  sensor.setPulseAmplitudeGreen(0);

  lcd.init(135, 240);
  lcd.setRotation(1);
  lcd.fillScreen(ST77XX_BLACK);
  lcd.setTextColor(ST77XX_GREEN);
  lcd.setTextSize(2);
  digitalWrite(LCD_BLK, HIGH);
}

bool autorizado() {
  HTTPClient http;
  http.begin(String(server) + "/api/autorizacion");
  int httpCode = http.GET();
  if (httpCode == 200) {
    String body = http.getString();
    bool autorizado = body.indexOf("true") > 0;
    http.end();
    return autorizado;
  }
  http.end();
  return false;
}

void loop() {
  long ir = sensor.getIR();
  long red = sensor.getRed();

  irMuestras[index_muestra] = ir;
  redMuestras[index_muestra] = red;
  index_muestra = (index_muestra + 1) % MUESTRAS;

  long sumaIR = 0, sumaRED = 0;
  for (int i = 0; i < MUESTRAS; i++) {
    sumaIR += irMuestras[i];
    sumaRED += redMuestras[i];
  }

  long avgIR = sumaIR / MUESTRAS;
  long avgRED = sumaRED / MUESTRAS;

  // HR
  if (ir > 10000 && (millis() - lastBeat) > 300) {
    unsigned long delta = millis() - lastBeat;
    lastBeat = millis();
    beatsPerMinute = 60.0 / (delta / 1000.0);
    if (beatsPerMinute > 30 && beatsPerMinute < 180) {
      beatAvg = (int)beatsPerMinute;
    }
  }

  // SpO2 con fórmula AC/DC
  float dcIR = avgIR;
  float dcRED = avgRED;
  float acIR = ir - dcIR;
  float acRED = red - dcRED;
  float ratio = (acRED / dcRED) / (acIR / dcIR);
  spo2 = constrain(110 - 25 * ratio, 70, 100);

  lcd.fillScreen(ST77XX_BLACK);
  lcd.setCursor(0, 20);
  lcd.print("HR: ");
  lcd.print(beatAvg);
  lcd.setCursor(0, 50);
  lcd.print("SpO2: ");
  lcd.print(spo2);
  lcd.print("%");
  lcd.setCursor(0, 90);
  lcd.setTextSize(1);
  lcd.print("IR avg: ");
  lcd.print(avgIR);
  lcd.setCursor(0, 110);
  lcd.print("RED avg: ");
  lcd.print(avgRED);
  lcd.setTextSize(2);

  if (autorizado() && avgIR > 10000 && beatAvg > 0) {
    HTTPClient http;
    http.begin(String(server) + "/api/datos");
    http.addHeader("Content-Type", "application/json");

    String payload = "{"hr":" + String(beatAvg) +
                     ","spo2":" + String(spo2) +
                     ","ir":" + String(avgIR) +
                     ","red":" + String(avgRED) + "}";

    int code = http.POST(payload);
    if (code > 0) {
      Serial.println("📡 Servidor: " + http.getString());
    } else {
      Serial.println("❌ Error HTTP");
    }
    http.end();
  }

  delay(500);
}
