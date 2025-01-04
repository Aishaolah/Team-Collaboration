#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#define N_INPUTS 3 // Weather, Soil, Resources

typedef struct {
    double weights[N_INPUTS];
    double bias;
} Perceptron;

#endif // PERCEPTRON_H
