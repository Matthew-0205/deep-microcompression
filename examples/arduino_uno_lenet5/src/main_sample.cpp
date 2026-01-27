// // #include <Arduino.h>
// // #include "deep_microcompression.h"

// // // Include your generated model artifacts
// // #include "uno_model.h"
// // #include "uno_model_test_input.h"

// // #define AVERAGING_TIME 1
// // #define INPUT_SIZE (1*28*28)
// // #define NUM_CLASSES 10

// // // --- Global Pointers ---
// // // We declare pointers globally, but assign them in setup()
// // // because the model instance is created in uno_model_def.cpp
// // int8_t* input_ptr;
// // int8_t* output_ptr;

// // void setup() {
// //   Serial.begin(9600);
// //   while (!Serial);

// //   Serial.print(F("\n\n--- DMC Arduino Uno Inference ---\n"));

// //   // 2. Link Pointers to Model Buffers
// //   // The 'uno_model' object is instantiated in src/uno_model_def.cpp
// //   input_ptr = uno_model.input;
// //   output_ptr = uno_model.output;

// //   Serial.print(F("Model Initialized. RAM usage check suggested.\n"));
// // }

// // void loop() {
// //   uint32_t t0;
// //   uint32_t dt;

// //   Serial.print(F("Loading Input Data...\n"));

// //   // Load Data (Flash -> RAM)
// //   // We use 'set_packed_value' to handle cases where input is bit-packed (e.g. 4-bit images)
// //   // This unpacks 'test_input' (from header) into the working 'input_ptr' buffer.
// //   for (int j = 0; j < INPUT_SIZE; j++) {
// //       int val = par_read_packed_intb(test_input, j);
// //       act_write_packed_intb(input_ptr, j, val);
// //   }

// //   Serial.print(F("Running Inference...\n"));
// //   t0 = millis();

// //   // Run Model
// //   for (int t = 0; t < AVERAGING_TIME; t++) {
// //     uno_model.predict();
// //   }

// //   dt = millis() - t0;

// //   // Report Results
// //   Serial.print(F("Inference Time: ")); Serial.print(dt / AVERAGING_TIME); Serial.println(F(" ms"));

// //   Serial.print(F("Predictions: "));
// //   for(int i=0; i < NUM_CLASSES; i++) {
// //       // Decode output (unpacks 4-bit/2-bit to int if needed)
// //       int val = (int)act_read_packed_intb(output_ptr, i);
// //       Serial.print(val); Serial.print(" ");
// //   }
// //   Serial.print(F("\n-----------------------------\n\n"));

// //   delay(2000);
// // }

// #if defined(__clang__)
// extern "C" double roundf(double __x) { return round(__x); }
// extern "C" double ceilf(double __x) { return ceil(__x); }
// #endif
// extern "C" void __cxa_pure_virtual()
// {
//   while (1)
//     ;
// }
// #include <Arduino.h>

// #include <avr/io.h>
// #include <util/delay.h>

// // On the Arduino Uno, the onboard LED is connected to PB5
// #define LED_PIN5 5
// #define LED_PIN4 4
// #define LED_PIN3 3
// #define LED_PIN2 2
// #define LED_PIN1 1
// #define LED_PIN0 0

// #include "deep_microcompression.h"

// // Include your generated model artifacts
// #include "lenet5_model.h"
// // Stores a sample image in pack c array
// #include "lenet5_model_test_input.h" 

// #define AVERAGING_TIME 1
// #define INPUT_SIZE (1 * 28 * 28)
// #define NUM_CLASSES 10

// // --- Global Pointers ---
// // We declare pointers globally, but assign them in setup()
// // because the model instance is created in uno_model_def.cpp
// int8_t *input_ptr;
// int8_t *output_ptr;
// extern char *__brkval;
// extern char __bss_end;

// void paint_stack()
// {
//   char *p = &__bss_end;
//   while (p < (char *)&p)
//   {
//     *p++ = 0x55;
//   }
// }

// uint16_t get_free_stack()
// {
//   char *p = &__bss_end;
//   uint16_t count = 0;
//   while (*p == 0x55 && p < (char *)&p)
//   {
//     count++;
//     p++;
//   }
//   return count;
// }
// void setup()
// {
//   paint_stack();
//   //   Serial.begin(9600);
//   //   while (!Serial);
//   // Serial.print(F("Free Stack before model init: "));
//   // Serial.println(get_free_stack());
//   // Serial.print(F("\n\n--- DMC Arduino Uno Inference ---\n"));

//   // 2. Link Pointers to Model Buffers
//   // The 'uno_model' object is instantiated in src/uno_model_def.cpp
//   // input_ptr = uno_model.input;
//   // output_ptr = uno_model.output;
//   //
//   // Serial.print(F("Model Initialized. RAM usage check suggested.\n"));
//   for (int j = 0; j < INPUT_SIZE; j++) {
//       int8_t val = parameter_read_packed_int4((int8_t*)test_input, j); 
//       lenet5_model.set_input(j, val);
//   }

//  // Run Model
//   for (int t = 0; t < AVERAGING_TIME; t++) {
//     lenet5_model.predict();
//   }
//   //   Serial.print(F("Free Stack after 1 model init: "));
//   // Serial.println(get_free_stack());
// }


// void loop()
// {
//   // uint32_t t0;
//   // uint32_t dt;

//   // Serial.print(F("Loading Input Data...\n"));

//   // Load Data (Flash -> RAM)
//   // We use 'set_packed_value' to handle cases where input is bit-packed (e.g. 4-bit images)
//   // This unpacks 'test_input' (from header) into the working 'input_ptr' buffer.
//   // for (int j = 0; j < INPUT_SIZE; j++)
//   // {
//   //   int val = par_read_packed_intb(test_input, j);
//   //   act_write_packed_intb(input_ptr, j, val);
//   // }

//   // // Serial.print(F("Running Inference...\n"));
//   // // t0 = millis();

//   // // Run Model
//   // for (int t = 0; t < AVERAGING_TIME; t++)
//   // {
//   //   uno_model.predict();
//   // }

//   // // dt = millis() - t0;

//   // Report Results
//   // Serial.print(F("Inference Time: ")); Serial.print(dt / AVERAGING_TIME); Serial.println(F(" ms"));
//   int a = -100000;
//   float val = 5; //safe_read_float(&DUCKS);
//   int id = 0;
//   // Serial.print(F("Predictions: "));
//   for (int i = 0; i < NUM_CLASSES; i++)
//   {
//     // Decode output (unpacks 4-bit/2-bit to int if needed)
//     // int val = (int)act_read_packed_intb(output_ptr, i);
//     if (val > a)
//     {
//       a = val;
//       id = i;
//     }
//     // Serial.print(val); Serial.print(" ");
//   }
//   PINB = (1 << id);

//   _delay_ms(val * 5);
//   PINB = (1 << id);

//   // _delay_ms(5000);

//   // Serial.print(F("\n-----------------------------\n\n"));

//   // delay(2000);
//   //     Serial.print(F("Free Stack after model init: "));
//   // Serial.println(get_free_stack());
// }

// int main(void)
// {
//   // Set PB5 as an output
//   DDRB |= (1 << LED_PIN5);
//   DDRB |= (1 << LED_PIN4);
//   DDRB |= (1 << LED_PIN3);
//   DDRB |= (1 << LED_PIN2);
//   DDRB |= (1 << LED_PIN1);
//   DDRB |= (1 << LED_PIN0);

//   while (0)
//   {
//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN5);
//     // Wait 500ms
//     _delay_ms(5000);

//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN5);

//     // Wait 500ms
//     _delay_ms(5000);
//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN4);
//     // Wait 500ms
//     _delay_ms(500);

//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN4);

//     // Wait 500ms
//     _delay_ms(500);
//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN3);
//     // Wait 500ms
//     _delay_ms(500);

//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN3);

//     // Wait 500ms
//     _delay_ms(500);

//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN2);
//     // Wait 500ms
//     _delay_ms(500);

//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN2);

//     // Wait 500ms
//     _delay_ms(500);

//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN1);
//     // Wait 500ms
//     _delay_ms(500);

//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN1);

//     // Wait 500ms
//     _delay_ms(500);

//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN0);
//     // Wait 500ms
//     _delay_ms(500);

//     // Toggle PB5 using the PINB register (writing 1 to a PIN bit toggles the PORT bit)
//     PINB = (1 << LED_PIN0);

//     // Wait 500ms
//     _delay_ms(500);
//   }
//   setup();

//   for (;;)
//   {
//     loop();
//   }

//   return 0;
// }
