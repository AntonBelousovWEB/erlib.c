#ifndef _SIMPLE_NEURAL_NETWORKS_H
#define _SIMPLE_NEURAL_NETWORKS_H

void multiple_input_multiple_output_nn(double * input_vector, 
                                       int INPUT_LEN, 
                                       double * output_vector, 
                                       int OUTPUT_LEN, 
                                       double weight_matrix[OUTPUT_LEN][INPUT_LEN]);

void brute_force_learning(int INPUT_LEN,
                          double input[INPUT_LEN],
                          int OUTPUT_LEN, 
                          double weight[OUTPUT_LEN][INPUT_LEN],
                          double expected_value,
                          double step_amount,
                          int itr);

#endif // _SIMPLE_NEURAL_NETWORKS_H