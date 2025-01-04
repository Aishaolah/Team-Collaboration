//Activation function (step function)
int step_function(double net_input) {
    return (net_input >= 0) ? 1 : 0;
}

//Calculate the net input (weighted sum of inputs)
double net_input(const Perceptron *perceptron, const double *inputs) {
    double sum = 0.0;
    for (int i = 0; i < NUM_INPUTS; ++i) {
        sum += perceptron->weights[i] * inputs[i];
    }
    sum += perceptron->bias;
    return sum;
}