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
