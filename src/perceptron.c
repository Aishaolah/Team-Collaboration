#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

//Number of input features: Weather, Resources
#define NUM_INPUTS 2
//Number of data samples (optional)
#define NUM_SAMPLES 100 

typedef struct {
    double weights[NUM_INPUTS];
    double bias;
} Perceptron;
    
//Activation function (step function)
int step_function(double net_input) {
    return (net_input >= 0) ? 1 : 0;
}

//Calculate the net input: weighted sum of inputs
double net_input(const Perceptron *perceptron, const double *inputs) {
    double sum = 0.0;
    for (int i = 0; i < NUM_INPUTS; ++i) {
        sum += perceptron->weights[i] * inputs[i];
    }
    sum += perceptron->bias;
    return sum;
}
//Make a prediction
int predict(const Perceptron *perceptron, const double *inputs) {
    double net = net_input(perceptron, inputs);
    return step_function(net);
}

//Train the Perceptron
void train_perceptron(Perceptron *perceptron,
                      const double inputs[][NUM_INPUTS],
                      const int *targets,
                      size_t num_samples,
                      double learning_rate,
                      int max_epochs) {
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        for (size_t i = 0; i < num_samples; ++i) {
            int prediction = predict(perceptron, inputs[i]);
            double error = targets[i] - prediction;

            //Update weights and bias
            for (int j = 0; j < NUM_INPUTS; ++j) {
                perceptron->weights[j] += learning_rate * error * inputs[i][j];
            }
            perceptron->bias += learning_rate * error;
        }
    }}
//Initialize the Perceptron with random weights and bias
void initialize_perceptron(Perceptron *perceptron) {
    srand(time(NULL));  //Seed the random number generator

    for (int i = 0; i < NUM_INPUTS; ++i) {
        perceptron->weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    perceptron->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}
