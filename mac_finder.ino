#include <WiFi.h>

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.print("MAC: ");
  Serial.println(WiFi.macAddress()); 
}
//HUSEYIN MAC : 94:52:c5:b0:8a:34
//SAFFET MAC : 94:54:c5:b5:ea:10
//Cafer MAC : 94:54:c5:b0:b1:24
//HAYDAR MAC: 94:54:c5:b2:48:e4
void loop() { delay(1000); }
