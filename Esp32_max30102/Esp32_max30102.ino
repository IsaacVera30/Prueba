#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include "MAX30105.h"
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>

#define LCD_CS   15
#define LCD_DC   2
#define LCD_RST  4
#define LCD_BLK  32
#define MUESTRAS 10

const char* ssid = "Extence";
const char* password = "@Isaac20";
const char* serverName = "http://192.168.0.102:5000/api/datos";

MAX30105 sensor;
Adafruit_ST7789 lcd = Adafruit_ST7789(LCD_CS, LCD_DC, LCD_RST);

int irMuestras[MUESTRAS], redMuestras[MUESTRAS];
int indexMuestra = 0;
unsigned long lastBeat = 0;
float beatsPerMinute;
int beatAvg = 0;

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi conectado");

  if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("❌ Sensor MAX30105 no detectado.");
    while (1);
  }
  sensor.setup();
  sensor.setPulseAmplitudeRed(0x3F);
  sensor.setPulseAmplitudeGreen(0);

  lcd.init(135, 240);
  lcd.setRotation(1);
  lcd.fillScreen(ST77XX_BLACK);
  lcd.setTextColor(ST77XX_YELLOW);
  lcd.setTextSize(2);
}

void loop() {
  long ir = sensor.getIR();
  long red = sensor.getRed();

  // Guardar muestras circulares
  irMuestras[indexMuestra] = ir;
  redMuestras[indexMuestra] = red;
  indexMuestra = (indexMuestra + 1) % MUESTRAS;

  // HR
  if (ir > 10000 && (millis() - lastBeat) > 300) {
    unsigned long delta = millis() - lastBeat;
    lastBeat = millis();
    beatsPerMinute = 60.0 / (delta / 1000.0);
    if (beatsPerMinute > 30 && beatsPerMinute < 180) {
      beatAvg = (int)beatsPerMinute;
    }
  }

  // Cálculo de SpO2 real con fórmula AC/DC
  float avgIR = 0, avgRED = 0;
  for (int i = 0; i < MUESTRAS; i++) {
    avgIR += irMuestras[i];
    avgRED += redMuestras[i];
  }
  avgIR /= MUESTRAS;
  avgRED /= MUESTRAS;

  float dcIR = avgIR;
  float dcRED = avgRED;
  float acIR = ir - dcIR;
  float acRED = red - dcRED;

  float ratio = (acRED / dcRED) / (acIR / dcIR);
  float spo2 = 110.0 - 25.0 * ratio;
  spo2 = constrain(spo2, 70, 100);

  // Mostrar en LCD
  lcd.fillScreen(ST77XX_BLACK);
  lcd.setCursor(0, 0);
  lcd.printf("IR avg: %ld\n", (long)avgIR);
  lcd.printf("RED avg: %ld\n", (long)avgRED);
  lcd.printf("HR: %d bpm\n", beatAvg);
  lcd.printf("SpO2: %.0f %%\n", spo2);

  // Enviar al servidor
  if (WiFi.status() == WL_CONNECTED && beatAvg > 0) {
    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    String payload = "{\"hr\":" + String(beatAvg) +
                     ",\"spo2\":" + String((int)spo2) +
                     ",\"ir\":" + String((int)avgIR) +
                     ",\"red\":" + String((int)avgRED) + "}";

    int code = http.POST(payload);
    if (code == 200) {
      String response = http.getString();
      Serial.println(response);

      // Extraer SYS y DIA de la respuesta
      int sysIndex = response.indexOf("sys");
      int diaIndex = response.indexOf("dia");
      if (sysIndex > 0 && diaIndex > 0) {
        float sys = response.substring(sysIndex + 5, response.indexOf(",", sysIndex)).toFloat();
        float dia = response.substring(diaIndex + 5, response.indexOf("}", diaIndex)).toFloat();

        lcd.setTextColor(ST77XX_GREEN);
        lcd.setCursor(0, 100);
        lcd.printf("SYS: %.0f mmHg\n", sys);
        lcd.printf("DIA: %.0f mmHg\n", dia);
      }
    }
    http.end();
  }

  delay(500);
}
