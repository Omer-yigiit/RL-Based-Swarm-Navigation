int ledPin = 2;   // ESP32 üzerinde built-in LED genelde GPIO2'dir
int count = 0;

void setup() {
  pinMode(ledPin, OUTPUT);
}

void loop() {
  if (count < 6) {
    digitalWrite(ledPin, HIGH);  // LED aç
    delay(1000);                 // 1 saniye bekle
    digitalWrite(ledPin, LOW);   // LED kapat
    delay(1000);                 // 1 saniye bekle
    count++;                     // Döngü sayısını arttır
  }
  else {
    // 3 defa yanıp söndükten sonra LED kapalı kalır
    digitalWrite(ledPin, LOW);
    while (true) {
      // Burada kalır, LED artık yanmaz
    }
  }
}
