/**************************************************************
  Fish Tank Monitor (Nano 33 IoT + Blynk + SD + TFT)

/******************** BLYNK ********************/
#define BLYNK_TEMPLATE_ID   "TMPL5a0bIoWiX"
#define BLYNK_TEMPLATE_NAME "DS18B20 water temperature sensor"
#define BLYNK_AUTH_TOKEN    "CgJYPhmQm9fCXxpe_B3nzNpdtHqlKE6p"

/******************** WIFI / BLYNK ********************/
#include <WiFiNINA.h>
#include <WiFiUdp.h>
#include <BlynkSimpleWiFiNINA.h>

/******************** SENSORS ********************/
#include <OneWire.h>
#include <DallasTemperature.h>
#include <math.h>

/******************** DISPLAY / INPUT / SD ********************/
#include <SPI.h>
#include <Encoder.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <Fonts/FreeMonoOblique9pt7b.h>
#include <SD.h>

/******************** WIFI CREDS ********************/
char ssid[] = "Glide_Resident";
char pass[] = "PlowStankPoppy";

/******************** TFT ********************/
#define TFT_CS  10
#define TFT_RST 8
#define TFT_DC  9
Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);

/******************** ENCODER ********************/
#define ENC_A   2
#define ENC_B   3
#define ENC_BTN 4
Encoder encoder(ENC_A, ENC_B);

/******************** SD ********************/
#define SD_CS 5
bool sdOK = false;

/******************** TEMPERATURE ********************/
#define ONE_WIRE_BUS 7
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

/******************** pH SENSOR ********************/
#define PH_PIN A1
const float ADC_REF_V = 3.3f;
const int   ADC_MAX   = 1023;
float neutralVoltage = 1.135f;
float acidSlope      = -6.32f;

/******************** WATER LEVEL ********************/
#define WATER_POT_PIN A4
const float h_cm[]  = {11, 10, 9, 8, 7, 6};
const int   r_adc[] = {876, 900, 910, 935, 949, 1022};
const int WN = sizeof(h_cm) / sizeof(h_cm[0]);

/******************** TIMERS ********************/
BlynkTimer timer;

/******************** THRESHOLDS (FROM BLYNK SLIDERS) ********************/
// Defaults used until Blynk sync happens after connect
float minTemp = 0;       // V2
float maxTemp = 50;      // V3
float minPH   = 6.0;     // V4
float maxPH   = 8.0;     // V5
float minWaterCm = 7.0;  // V7 

/******************** BLYNK SLIDER HANDLERS ********************/
BLYNK_WRITE(V2) { minTemp = param.asFloat(); }
BLYNK_WRITE(V3) { maxTemp = param.asFloat(); }
BLYNK_WRITE(V4) { minPH   = param.asFloat(); }
BLYNK_WRITE(V5) { maxPH   = param.asFloat(); }
BLYNK_WRITE(V7) { minWaterCm = param.asFloat(); }   // Min water level slider

/******************** SYNC SLIDERS ON CONNECT ********************/
BLYNK_CONNECTED() {
  Blynk.syncVirtual(V2);
  Blynk.syncVirtual(V3);
  Blynk.syncVirtual(V4);
  Blynk.syncVirtual(V5);
  Blynk.syncVirtual(V7);
}

/******************** BLYNK ALERTS (EVENTS) ********************/
const char* EVT_TEMP_LOW   = "temp_low";
const char* EVT_TEMP_HIGH  = "temp_high";
const char* EVT_PH_LOW     = "ph_low";
const char* EVT_PH_HIGH    = "ph_high";
const char* EVT_WATER_LOW  = "water_level_low";

// Cooldown to avoid spam (10 minutes)
const unsigned long ALERT_COOLDOWN_MS = 10UL * 60UL * 1000UL;
unsigned long lastTempLowAlertMs  = 0;
unsigned long lastTempHighAlertMs = 0;
unsigned long lastPhLowAlertMs    = 0;
unsigned long lastPhHighAlertMs   = 0;
unsigned long lastWaterLowAlertMs = 0;

void maybeSendAlert(const char* eventCode, const char* msg, unsigned long& lastMs) {
  if (!Blynk.connected()) return;

  unsigned long now = millis();
  if (now - lastMs < ALERT_COOLDOWN_MS) return;

  lastMs = now;
  Blynk.logEvent(eventCode, msg);

  Serial.print("ALERT ");
  Serial.print(eventCode);
  Serial.print(": ");
  Serial.println(msg);
}

/******************** MENU ********************/
const char* menuItems[] = { "Data", "Settings", "Save", "Reset" };
const int menuCount = 4;

const char* dataMenu[]     = { "View", "Export", "Back" };
const char* settingsMenu[] = { "Change values", "Calibrate", "Back" };
const char* saveMenu[]     = { "Save to SD", "Back" };

enum MenuLevel { MENU_MAIN, MENU_SUB };
MenuLevel menuLevel = MENU_MAIN;

const char** currentMenu = nullptr;
int currentMenuCount = 0;

int selectedItem = 0;
long lastEncoderPos = 0;

const int ROW_HEIGHT   = 32;
const int ROW_Y_OFFSET = 15;

int16_t tbx, tby;
uint16_t tbw, tbh;

enum ScreenMode { SCREEN_MENU, SCREEN_VIEW_VALUES };
ScreenMode screenMode = SCREEN_MENU;

/******************** SENSOR STATE (30s averages) ********************/
float lastTempC = NAN;
float lastPH    = NAN;
float lastWaterCm = NAN;
int   lastWaterAvgAdc = -1;

bool tempReady  = false;
bool phReady    = false;
bool waterReady = false;

/******************** 30s AVERAGING WINDOWS ********************/
const unsigned long AVG_WINDOW_MS = 30000;

// Water
const unsigned long WATER_SAMPLE_INTERVAL_MS = 50;
unsigned long waterLastSampleMs  = 0;
unsigned long waterWindowStartMs = 0;
long waterAdcSum = 0;
unsigned long waterSampleCount = 0;

// Temp
const unsigned long TEMP_CONV_MS = 800;
const unsigned long TEMP_SAMPLE_INTERVAL_MS = 2000;
unsigned long tempLastKickMs  = 0;
bool tempConversionRunning = false;
unsigned long tempWindowStartMs = 0;
double tempSum = 0;
unsigned long tempCount = 0;

// pH
const unsigned long PH_SAMPLE_INTERVAL_MS = 3000;
unsigned long phLastSampleMs  = 0;
unsigned long phWindowStartMs = 0;
double phSum = 0;
unsigned long phCount = 0;

/******************** SPI SAFETY ********************/
inline void deselectAllSPI() {
  digitalWrite(TFT_CS, HIGH);
  digitalWrite(SD_CS, HIGH);
}

/******************** UI HELPERS ********************/
void drawStatusLine(const char* msg) {
  tft.fillRect(0, 145, 160, 15, ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE, ST77XX_BLACK);
  tft.setCursor(2, 156);
  tft.print(msg);
}

int getMenuCount() {
  return (menuLevel == MENU_MAIN) ? menuCount : currentMenuCount;
}

const char* getMenuItem(int i) {
  return (menuLevel == MENU_MAIN) ? menuItems[i] : currentMenu[i];
}

void drawMenuItem(int i, bool selected) {
  int baselineY = i * ROW_HEIGHT + ROW_Y_OFFSET;

  tft.fillRect(
    0,
    baselineY + tby - 2,
    160,
    tbh + 6,
    selected ? ST77XX_CYAN : ST77XX_BLACK);

  tft.setTextColor(selected ? ST77XX_MAGENTA : ST77XX_WHITE);
  tft.setCursor(10, baselineY);
  tft.print(getMenuItem(i));
}

void drawMenuFull() {
  tft.fillScreen(ST77XX_BLACK);
  int count = getMenuCount();
  for (int i = 0; i < count; i++) {
    drawMenuItem(i, i == selectedItem);
  }
}

/******************** WATER LEVEL HELPERS ********************/
float interpHeightWater(int adc) {
  if (adc <= r_adc[0]) return h_cm[0];
  if (adc >= r_adc[WN - 1]) return h_cm[WN - 1];

  for (int i = 0; i < WN - 1; i++) {
    int r1 = r_adc[i];
    int r2 = r_adc[i + 1];
    if (adc >= min(r1, r2) && adc <= max(r1, r2)) {
      float h1 = h_cm[i];
      float h2 = h_cm[i + 1];
      float t  = (float)(adc - r1) / (float)(r2 - r1);
      return h1 + t * (h2 - h1);
    }
  }
  return h_cm[WN - 1];
}

float adcToHeightStableWater(int adc) {
  if (adc >= r_adc[WN - 1]) return 6.0f;
  if (adc <= 888) return 11.0f;
  if (adc <= 905) return 10.0f;
  if (adc <= 923) return  9.0f;
  return interpHeightWater(adc);
}

/******************** pH READ ********************/
float readVoltageAvg(int n = 20) {
  long sum = 0;
  for (int i = 0; i < n; i++) {
    sum += analogRead(PH_PIN);
    delayMicroseconds(300);
  }
  return ((float)sum / n) * ADC_REF_V / ADC_MAX;
}

/******************** WIFI / BLYNK RECONNECT ********************/
unsigned long lastWiFiAttemptMs  = 0;
unsigned long lastBlynkAttemptMs = 0;
const unsigned long WIFI_RETRY_MS  = 8000;
const unsigned long BLYNK_RETRY_MS = 5000;

void connectWiFiIfNeeded() {
  if (WiFi.status() == WL_CONNECTED) return;

  unsigned long now = millis();
  if (now - lastWiFiAttemptMs < WIFI_RETRY_MS) return;
  lastWiFiAttemptMs = now;

  drawStatusLine("WiFi: connecting");
  Serial.println("Connecting WiFi...");
  WiFi.begin(ssid, pass);
}

void connectBlynkIfNeeded() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (Blynk.connected()) return;

  unsigned long now = millis();
  if (now - lastBlynkAttemptMs < BLYNK_RETRY_MS) return;
  lastBlynkAttemptMs = now;

  drawStatusLine("Blynk: connecting");
  Serial.println("Connecting Blynk...");
  Blynk.config(BLYNK_AUTH_TOKEN);
  bool ok = Blynk.connect(8000);

  if (ok) drawStatusLine("Blynk: OK");
  else    drawStatusLine("Blynk: FAIL");
}

/******************** SENSOR UPDATES ********************/
void updateWaterAverage() {
  unsigned long now = millis();

  if (now - waterLastSampleMs >= WATER_SAMPLE_INTERVAL_MS) {
    waterLastSampleMs = now;
    int raw = analogRead(WATER_POT_PIN);
    waterAdcSum += raw;
    waterSampleCount++;
  }

  if (now - waterWindowStartMs >= AVG_WINDOW_MS) {
    if (waterSampleCount > 0) {
      lastWaterAvgAdc = (int)(waterAdcSum / (long)waterSampleCount);
      lastWaterCm = adcToHeightStableWater(lastWaterAvgAdc);
      waterReady = true;

      Serial.print("WATER 30s AVG ADC = ");
      Serial.print(lastWaterAvgAdc);
      Serial.print("  CM = ");
      Serial.println(lastWaterCm, 2);

      if (Blynk.connected()) Blynk.virtualWrite(V6, lastWaterCm);

      // LOW water alert (uses slider V7 threshold)
      if (lastWaterCm < minWaterCm) {
        char msg[80];
        snprintf(msg, sizeof(msg), "Water LOW: %.2fcm (< %.2fcm)", lastWaterCm, minWaterCm);
        maybeSendAlert(EVT_WATER_LOW, msg, lastWaterLowAlertMs);
      }
    }
    waterAdcSum = 0;
    waterSampleCount = 0;
    waterWindowStartMs = now;
  }
}

void updateTempAverage() {
  unsigned long now = millis();

  if (!tempConversionRunning && (now - tempLastKickMs >= TEMP_SAMPLE_INTERVAL_MS)) {
    tempLastKickMs = now;
    sensors.setWaitForConversion(false);
    sensors.requestTemperatures();
    tempConversionRunning = true;
    return;
  }

  if (tempConversionRunning && (now - tempLastKickMs >= TEMP_CONV_MS)) {
    tempConversionRunning = false;

    float tC = sensors.getTempCByIndex(0);

    Serial.print("TEMP raw = ");
    Serial.println(tC, 2);

    if (tC == DEVICE_DISCONNECTED_C || tC <= -100 || tC >= 126 || fabs(tC - 85.0) < 0.01) {
      Serial.println("TEMP invalid (wiring/pullup/too-early read)");
      return;
    }

    tempSum += tC;
    tempCount++;
  }

  if (now - tempWindowStartMs >= AVG_WINDOW_MS) {
    if (tempCount > 0) {
      lastTempC = (float)(tempSum / (double)tempCount);
      tempReady = true;

      Serial.print("TEMP 30s AVG = ");
      Serial.println(lastTempC, 2);

      if (Blynk.connected()) Blynk.virtualWrite(V0, lastTempC);

      // Temperature alerts using slider thresholds V2/V3
      if (lastTempC < minTemp) {
        char msg[80];
        snprintf(msg, sizeof(msg), "Temp LOW: %.2fC (< %.2fC)", lastTempC, minTemp);
        maybeSendAlert(EVT_TEMP_LOW, msg, lastTempLowAlertMs);
      } else if (lastTempC > maxTemp) {
        char msg[80];
        snprintf(msg, sizeof(msg), "Temp HIGH: %.2fC (> %.2fC)", lastTempC, maxTemp);
        maybeSendAlert(EVT_TEMP_HIGH, msg, lastTempHighAlertMs);
      }
    } else {
      Serial.println("TEMP window had 0 valid samples");
    }

    tempSum = 0;
    tempCount = 0;
    tempWindowStartMs = now;
  }
}

void updatePHAverage() {
  unsigned long now = millis();

  if (now - phLastSampleMs >= PH_SAMPLE_INTERVAL_MS) {
    phLastSampleMs = now;

    float v = readVoltageAvg();
    float p = 7.0f + (v - neutralVoltage) * acidSlope;

    Serial.print("PH ADC = ");
    Serial.print(analogRead(PH_PIN));
    Serial.print("  volts=");
    Serial.print(v, 3);
    Serial.print("  pH=");
    Serial.println(p, 2);

    if (p > 0 && p < 14) {
      phSum += p;
      phCount++;
    }
  }

  if (now - phWindowStartMs >= AVG_WINDOW_MS) {
    if (phCount > 0) {
      lastPH = (float)(phSum / (double)phCount);
      phReady = true;

      Serial.print("PH 30s AVG = ");
      Serial.println(lastPH, 2);

      if (Blynk.connected()) Blynk.virtualWrite(V1, lastPH);

      // pH alerts using slider thresholds V4/V5
      if (lastPH < minPH) {
        char msg[80];
        snprintf(msg, sizeof(msg), "pH LOW: %.2f (< %.2f)", lastPH, minPH);
        maybeSendAlert(EVT_PH_LOW, msg, lastPhLowAlertMs);
      } else if (lastPH > maxPH) {
        char msg[80];
        snprintf(msg, sizeof(msg), "pH HIGH: %.2f (> %.2f)", lastPH, maxPH);
        maybeSendAlert(EVT_PH_HIGH, msg, lastPhHighAlertMs);
      }
    } else {
      Serial.println("PH window had 0 valid samples");
    }

    phSum = 0;
    phCount = 0;
    phWindowStartMs = now;
  }
}

/******************** SD DIAGNOSTICS (COMPATIBLE) ********************/
void sdDiagnosticsSerial() {
  Serial.println("---- SD DIAGNOSTICS ----");
  if (!sdOK) {
    Serial.println("SD: init failed");
    Serial.println("If TFT works but SD fails: likely wiring/SPI pins/3.3V issue");
    Serial.println("------------------------");
    return;
  }

  File root = SD.open("/");
  if (!root) {
    Serial.println("SD: open / failed");
    Serial.println("------------------------");
    return;
  }

  Serial.println("SD: opened / OK. Listing files:");
  while (true) {
    File entry = root.openNextFile();
    if (!entry) break;
    Serial.print("  ");
    Serial.print(entry.name());
    if (!entry.isDirectory()) {
      Serial.print("  (");
      Serial.print(entry.size());
      Serial.println(" bytes)");
    } else {
      Serial.println("  <DIR>");
    }
    entry.close();
  }
  root.close();

  File f = SD.open("sd_test.txt", FILE_WRITE);
  if (f) {
    f.println("sd write test ok");
    f.close();
    Serial.println("SD: wrote sd_test.txt OK");
  } else {
    Serial.println("SD: could not create sd_test.txt");
  }

  Serial.println("------------------------");
}

/******************** SD LOGGING ********************/
void ensureHeader() {
  if (!SD.exists("tank.csv")) {
    File f = SD.open("tank.csv", FILE_WRITE);
    if (f) {
      f.println("tempC_avg30s,pH_avg30s,water_cm_avg30s,water_adc_avg30s");
      f.close();
    }
  }
}

void logToSD() {
  if (!sdOK) return;
  if (!tempReady || !phReady || !waterReady) return;

  deselectAllSPI();
  digitalWrite(SD_CS, LOW);
  delayMicroseconds(5);

  ensureHeader();

  File f = SD.open("tank.csv", FILE_WRITE);
  if (!f) {
    Serial.println("SD: open fail (tank.csv)");
    digitalWrite(SD_CS, HIGH);
    return;
  }

  f.print(lastTempC, 2);    f.print(",");
  f.print(lastPH, 2);       f.print(",");
  f.print(lastWaterCm, 2);  f.print(",");
  f.println(lastWaterAvgAdc);
  f.close();

  digitalWrite(SD_CS, HIGH);
  Serial.println("Logged to SD (tank.csv)");
}

void saveToSD_now() {
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE, ST77XX_BLACK);
  tft.setCursor(10, 40);

  if (!sdOK) {
    tft.print("SD not found");
    delay(700);
    drawMenuFull();
    return;
  }

  logToSD();

  tft.print("Saved to SD");
  delay(700);
  drawMenuFull();
}

/******************** VIEW SCREEN ********************/
void drawViewValuesScreen() {
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE, ST77XX_BLACK);

  tft.setCursor(5, 18);
  tft.print("VIEW VALUES (30s)");

  tft.setCursor(5, 50);
  tft.print("Temp: ");
  if (!tempReady) tft.print("WAIT 30s");
  else { tft.print(lastTempC, 1); tft.print(" C"); }

  tft.setCursor(5, 75);
  tft.print("pH:   ");
  if (!phReady) tft.print("WAIT 30s");
  else tft.print(lastPH, 2);

  tft.setCursor(5, 100);
  tft.print("Water:");
  if (!waterReady) tft.print(" WAIT 30s");
  else { tft.print(" "); tft.print(lastWaterCm, 1); tft.print(" cm"); }

  tft.setCursor(5, 125);
  tft.print("Press to Back");
}

/******************** MENU NAV ********************/
void enterSubMenu(int mainItem) {
  menuLevel = MENU_SUB;
  selectedItem = 0;
  lastEncoderPos = encoder.read() / 4;

  switch (mainItem) {
    case 0: currentMenu = dataMenu;     currentMenuCount = 3; break;
    case 1: currentMenu = settingsMenu; currentMenuCount = 3; break;
    case 2: currentMenu = saveMenu;     currentMenuCount = 2; break;
    default: currentMenu = dataMenu;    currentMenuCount = 3; break;
  }

  drawMenuFull();
}

void goBackToMain() {
  menuLevel = MENU_MAIN;
  selectedItem = 0;
  lastEncoderPos = encoder.read() / 4;
  drawMenuFull();
}

/******************** BUTTON ACTION ********************/
void handleSelect() {
  if (screenMode == SCREEN_VIEW_VALUES) {
    screenMode = SCREEN_MENU;
    drawMenuFull();
    return;
  }

  if (menuLevel == MENU_MAIN) {
    if (selectedItem == 3) {
      NVIC_SystemReset();
    } else {
      enterSubMenu(selectedItem);
    }
    return;
  }

  const char* item = getMenuItem(selectedItem);

  if (!strcmp(item, "Back")) {
    goBackToMain();
    return;
  }

  if (currentMenu == dataMenu) {
    if (!strcmp(item, "View")) {
      screenMode = SCREEN_VIEW_VALUES;
      drawViewValuesScreen();
      return;
    }
    if (!strcmp(item, "Export")) {
      saveToSD_now();
      return;
    }
  }

  if (currentMenu == saveMenu) {
    if (!strcmp(item, "Save to SD")) {
      saveToSD_now();
      return;
    }
  }

  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE, ST77XX_BLACK);
  tft.setCursor(10, 50);
  tft.print(item);
  delay(700);
  drawMenuFull();
}

/******************** SETUP ********************/
void setup() {
  pinMode(ENC_BTN, INPUT_PULLUP);

  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 3000) { }
  Serial.println("BOOT OK");

  pinMode(TFT_CS, OUTPUT);
  pinMode(SD_CS, OUTPUT);
  deselectAllSPI();

  SPI.begin();

  // SD INIT FIRST
  Serial.print("SD init on CS=");
  Serial.println(SD_CS);
  sdOK = SD.begin(SD_CS);
  Serial.println(sdOK ? "SD init OK" : "SD init failed");

  sdDiagnosticsSerial();

  // TFT AFTER
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(1);
  tft.setFont(&FreeMonoOblique9pt7b);
  tft.getTextBounds("Ag", 0, 0, &tbx, &tby, &tbw, &tbh);
  tft.fillScreen(ST77XX_BLACK);

  sensors.begin();
  Serial.print("DS18B20 count = ");
  Serial.println(sensors.getDeviceCount());
  sensors.setResolution(12);
  sensors.setWaitForConversion(false);

  unsigned long now = millis();
  waterWindowStartMs = now;
  tempWindowStartMs  = now;
  phWindowStartMs    = now;

  connectWiFiIfNeeded();
  connectBlynkIfNeeded();

  // Log to SD every 60 seconds
  timer.setInterval(60000, logToSD);

  screenMode = SCREEN_MENU;
  menuLevel = MENU_MAIN;
  selectedItem = 0;
  lastEncoderPos = encoder.read() / 4;
  drawMenuFull();

  drawStatusLine(sdOK ? "SD: OK" : "SD: FAIL");
}

/******************** LOOP ********************/
void loop() {
  connectWiFiIfNeeded();
  connectBlynkIfNeeded();
  if (Blynk.connected()) Blynk.run();
  timer.run();

  updateWaterAverage();
  updateTempAverage();
  updatePHAverage();

  static unsigned long lastViewRefreshMs = 0;
  if (screenMode == SCREEN_VIEW_VALUES) {
    if (millis() - lastViewRefreshMs > 500) {
      lastViewRefreshMs = millis();
      drawViewValuesScreen();
    }
  }

  if (screenMode == SCREEN_MENU) {
    long newPos = encoder.read() / 4;

    if (newPos != lastEncoderPos) {
      int prevItem = selectedItem;

      if (newPos > lastEncoderPos) selectedItem++;
      else selectedItem--;

      int count = getMenuCount();
      if (selectedItem < 0) selectedItem = count - 1;
      if (selectedItem >= count) selectedItem = 0;

      drawMenuItem(prevItem, false);
      drawMenuItem(selectedItem, true);

      lastEncoderPos = newPos;
    }
  }

  static bool lastBtn = HIGH;
  static unsigned long lastPressMs = 0;

  bool btn = digitalRead(ENC_BTN);
  if (lastBtn == HIGH && btn == LOW) {
    if (millis() - lastPressMs > 220) {
      handleSelect();
      lastPressMs = millis();
    }
  }
  lastBtn = btn;
}
