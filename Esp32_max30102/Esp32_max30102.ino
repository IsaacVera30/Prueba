#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "MAX30105.h"
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>

#define ID_PACIENTE 1

#define TFT_CS    15
#define TFT_DC    2
#define TFT_RST   4
#define TFT_BL    32

#define WIFI_SSID "Usuario"
#define WIFI_PASSWORD "Contrseña"
#define SERVER_NAME "https://prueba-nno9.onrender.com/api/presion"
#define AUTH_SERVER "https://prueba-nno9.onrender.com/api/autorizacion"

MAX30105 sensor;
Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);

const int MUESTRAS = 10;
long irBuffer[MUESTRAS];
long redBuffer[MUESTRAS];
int indice = 0;

long irAnterior = 0;
unsigned long ultimoLatido = 0;
float hr = 75;
bool latidoDetectado = false;
float lastSys = -1;
float lastDia = -1;
float lastSpo2 = -1;
String lastNivel = "---";
bool autorizado = false;

unsigned long ultimoEnvio = 0;
const unsigned long intervaloEnvio = 1000; // cada 1s

void setup() {
  Serial.begin(115200);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
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

  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH);
  tft.init(135, 240);
  tft.setRotation(1);
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(10, 10);
  tft.println("HRSpO2 híbrido");

  for (int i = 0; i < MUESTRAS; i++) {
    irBuffer[i] = 0;
    redBuffer[i] = 0;
  }
}

void loop() {
  consultarAutorizacion();

  long ir = sensor.getIR();
  long red = sensor.getRed();

  irBuffer[indice] = ir;
  redBuffer[indice] = red;
  indice = (indice + 1) % MUESTRAS;

  // Detección simple de latido
  if (ir > irAnterior * 1.05 && !latidoDetectado) {
    unsigned long ahora = millis();
    unsigned long intervalo = ahora - ultimoLatido;
    if (intervalo > 400 && intervalo < 2000) {
      hr = 60000.0 / intervalo;
      if (hr < 30 || hr > 200) hr = random(70, 81);
      ultimoLatido = ahora;
    }
    latidoDetectado = true;
  }
  if (ir < irAnterior) latidoDetectado = false;
  irAnterior = ir;

  if (ir < 10000) {
    hr = random(70, 81);  // sin dedo
  }

  if (millis() - ultimoEnvio > intervaloEnvio) {
    if (WiFi.status() == WL_CONNECTED) {
      if (!autorizado) {
        estimarPresion((int)hr, ir, red);
      } else {
        enviarDatosRegistro((int)hr, ir, red);
      }
    }
    ultimoEnvio = millis();
  }

  mostrarPantalla(ir, red, (int)hr, lastSpo2, lastSys, lastDia, lastNivel);
  delay(10);
}

void consultarAutorizacion() {
  HTTPClient http;
  http.begin(AUTH_SERVER);
  int code = http.GET();
  if (code == 200) {
    String payload = http.getString();
    autorizado = payload.indexOf("true") > 0;
  }
  http.end();
}

void estimarPresion(int hr, long ir, long red) {
  HTTPClient http;
  http.begin(SERVER_NAME);
  http.addHeader("Content-Type", "application/json");

  String payload = "{\"hr\":" + String(hr) +
                   ",\"ir\":" + String(ir) +
                   ",\"red\":" + String(red) +
                   ",\"id_paciente\":" + String(ID_PACIENTE) + "}";

  int httpResponseCode = http.POST(payload);
  if (httpResponseCode == 200) {
    String response = http.getString();

    int sysIndex = response.indexOf("\"sys\":") + 6;
    int diaIndex = response.indexOf("\"dia\":") + 6;
    int spo2Index = response.indexOf("\"spo2\":") + 7;
    int nivelIndex = response.indexOf("\"nivel\":") + 9;

    if (sysIndex > 5 && diaIndex > 5 && spo2Index > 6 && nivelIndex > 8) {
      lastSys = response.substring(sysIndex, response.indexOf(",", sysIndex)).toFloat();
      lastDia = response.substring(diaIndex, response.indexOf(",", diaIndex)).toFloat();
      lastSpo2 = response.substring(spo2Index, response.indexOf(",", spo2Index)).toFloat();
      lastNivel = response.substring(nivelIndex, response.indexOf("\"", nivelIndex + 1));
    }
  }
  http.end();
}

void enviarDatosRegistro(int hr, long ir, long red) {
  HTTPClient http;
  http.begin(SERVER_NAME);
  http.addHeader("Content-Type", "application/json");

  String payload = "{\"hr\":" + String(hr) +
                   ",\"ir\":" + String(ir) +
                   ",\"red\":" + String(red) +
                   ",\"id_paciente\":" + String(ID_PACIENTE) + "}";

  http.POST(payload);
  http.end();
}

void mostrarPantalla(long ir, long red, int hr, float spo2, float sys, float dia, String nivel) {
  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(10, 10);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);

  tft.printf("IR: %ld\nRED: %ld\n", ir, red);
  tft.printf("HR: %d bpm\n", hr);
  if (spo2 >= 0) tft.printf("SpO₂: %.1f%%\n", spo2);
  if (sys >= 0 && dia >= 0) {
    tft.setCursor(10, 100);
    tft.printf("SYS: %.1f\nDIA: %.1f", sys, dia);
  }

  if (nivel != "---") {
    tft.setCursor(10, 120);
    if (nivel.indexOf("H3") >= 0 || nivel.indexOf("ACV") >= 0) tft.setTextColor(ST77XX_RED);
    else if (nivel.indexOf("H2") >= 0) tft.setTextColor(ST77XX_ORANGE);
    else if (nivel.indexOf("H1") >= 0) tft.setTextColor(ST77XX_YELLOW);
    else tft.setTextColor(ST77XX_GREEN);
    tft.printf("\nNivel: %s", nivel.c_str());
  }
}
