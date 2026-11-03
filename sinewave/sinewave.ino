#include <math.h>

const int buttonPin = 2;   
const int analogOutPin = 9; 

// --- Waveform Configuration Constants ---
const float FREQUENCY_HZ = 50.0; // Desired sine wave frequency in Hertz
const int AMPLITUDE = 127;       // Peak amplitude (0 to 127 for 8-bit output)
const int DC_OFFSET = 127;       // Center of the wave (127 keeps it within 0-255)
const int SINE_STEPS = 100;      // Resolution of the wave

uint8_t sineTable[SINE_STEPS];
unsigned long stepDelayUs;

void setup() {
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(analogOutPin, OUTPUT);

  // Calculate required delay per step to hit the target frequency
  stepDelayUs = 1000000UL / (FREQUENCY_HZ * SINE_STEPS);

  // Pre-calculate the scaled sine wave values
  for (int i = 0; i < SINE_STEPS; i++) {
    sineTable[i] = (uint8_t)((sin(i * 2.0 * M_PI / SINE_STEPS) * AMPLITUDE) + DC_OFFSET);
  }
}

void loop() {
  if (digitalRead(buttonPin) == LOW) {
    for (int i = 0; i < SINE_STEPS; i++) {
      analogWrite(analogOutPin, sineTable[i]);
      delayMicroseconds(stepDelayUs); 
    }
  } else {
    analogWrite(analogOutPin, 0); 
  }
}
