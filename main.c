#include <stdio.h>
#include "./Headers/simple_neural_networks.h"

#define Sad 0.9
#define SAD_PREDICTION_IDX      0
#define SICK_PREDICTION_IDX     1
#define ACTIVE_PREDICTION_IDX   2

#define  IN_LEN  3
#define  OUT_LEN 3

double expected_value = 0.8;
double step_amount = 0.001;

double predicted_results[3];
                                // temp   hum   air_q 
double weights[OUT_LEN][IN_LEN] = {
                                   {-2,   9.5,  2.01}, // sad
                                   {-0.8, 7.2,  6.3},  // sick?
                                   {-0.5, 0.45, 0.9}   // active?
                                  };
double inputs[IN_LEN] = {30, 87, 110}; // temp hum air_q

int main() {
  brute_force_learning(IN_LEN, inputs, OUT_LEN, weights, expected_value, step_amount,800);
  multiple_input_multiple_output_nn(inputs, IN_LEN, predicted_results, OUT_LEN, weights);
  printf("\nSad prediction: %f\n\n", predicted_results[SAD_PREDICTION_IDX]);
  printf("\nSick prediction: %f\n\n", predicted_results[SICK_PREDICTION_IDX]);
  printf("\nActive prediction: %f\n\n", predicted_results[ACTIVE_PREDICTION_IDX]);
  return 0;
}