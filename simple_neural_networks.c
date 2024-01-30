#include <stdio.h>
#include <math.h>
#include "./Headers/simple_neural_networks.h"

void matrix_vector_multiply(double * input_vector, 
                            int INPUT_LEN, 
                            double * output_vector, 
                            int OUTPUT_LEN, 
                            double weight_matrix[OUTPUT_LEN][INPUT_LEN]) {

    for(int k = 0; k<OUTPUT_LEN; k++) {
        for(int i = 0; i<INPUT_LEN; i++) {
            output_vector[k] += input_vector[i] * weight_matrix[k][i];
        }
    }                            
}

void multiple_input_multiple_output_nn(double * input_vector, 
                                       int INPUT_LEN, 
                                       double * output_vector, 
                                       int OUTPUT_LEN, 
                                       double weight_matrix[OUTPUT_LEN][INPUT_LEN]) {

    matrix_vector_multiply(input_vector, INPUT_LEN, output_vector, OUTPUT_LEN, weight_matrix);
}

void brute_force_learning(int INPUT_LEN,
                          double input[INPUT_LEN],
                          int OUTPUT_LEN, 
                          double weight[OUTPUT_LEN][INPUT_LEN],
                          double expected_value,
                          double step_amount,
                          int itr) {

    double prediction, error;
    double up_prediction, up_error;
    double down_prediction, down_error;     

    for(int k = 0; k<OUTPUT_LEN; k++) {
        for(int i = 0; i<INPUT_LEN; i++) {
            for(int j = 0; j < itr; j++) {
                prediction = input[i]*weight[k][i];
                error = powf((prediction - expected_value), 2);
                printf(
                    "\n\nNumber: %d\nError: %f\n Prediction: %f\n\n", 
                    i, error, prediction
                );
                up_prediction = input[i] * (weight[k][i] + step_amount);
                up_error = powf((expected_value - up_prediction), 2);
    
                down_prediction = input[i] * (weight[k][i] - step_amount);
                down_error = powf((expected_value - down_prediction), 2);
    
                if(down_error < up_error)
                    weight[k][i] = weight[k][i] - step_amount;
                if(down_error > up_error) 
                    weight[k][i] = weight[k][i] + step_amount;
           }
        }
    } 
}