#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define NUM_INPUTS 2
#define NUM_SAMPLES 100

typedef struct {
    double weights[NUM_INPUTS];
    double bias;
} Perceptron;

void perceptron_init(Perceptron *p) {
    srand(time(NULL));
    for (int i = 0; i < NUM_INPUTS; ++i) {
        p->weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    p->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

int perceptron_classify(const Perceptron *p, const double *inputs) {
    double sum = p->bias;
    for (int i = 0; i < NUM_INPUTS; ++i) {
        sum += p->weights[i] * inputs[i];
    }
    return sum >= 0 ? 1 : 0;
}

void perceptron_updateWeights(Perceptron *p, const double *inputs, int target, double learning_rate) {
    int prediction = perceptron_classify(p, inputs);
    double error = target - prediction;

    if (error != 0) {
        for (int i = 0; i < NUM_INPUTS; ++i) {
            p->weights[i] += learning_rate * error * inputs[i];
        }
        p->bias += learning_rate * error;
    }
}

void generateCSV(const char *filename, int num_points) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Failed to create file: %s\n", filename);
        return;
    }

    fprintf(file, "x,y,label\n"); // Header row

    srand(time(NULL));

    for (int i = 0; i < num_points / 2; ++i) {
        double x = ((double)rand() / RAND_MAX) * 0.8 - 0.9;
        double y = ((double)rand() / RAND_MAX) * 0.8 - 0.9;
        fprintf(file, "%f,%f,0\n", x, y);
    }

    for (int i = 0; i < num_points / 2; ++i) {
        double x = ((double)rand() / RAND_MAX) * 0.8 + 0.1;
        double y = ((double)rand() / RAND_MAX) * 0.8 + 0.1;
        fprintf(file, "%f,%f,1\n", x, y);
    }

    fclose(file);
    printf("Dataset generated: %s\n", filename);
}

int loadDataset(const char *filename, double inputs[][NUM_INPUTS], int *targets, size_t num_samples) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Failed to open file: %s\n", filename);
        return 0;
    }

    char line[256];
    size_t count = 0;
    fgets(line, sizeof(line), file); // Skip header

    while (fgets(line, sizeof(line), file) && count < num_samples) {
        sscanf(line, "%lf,%lf,%d", &inputs[count][0], &inputs[count][1], &targets[count]);
        ++count;
    }

    fclose(file);
    return 1;
}

void visualize(SDL_Renderer *renderer, double inputs[][NUM_INPUTS], int *targets, size_t num_samples, const Perceptron *perceptron) {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);

    // Draw data points
    for (size_t i = 0; i < num_samples; ++i) {
        int screenX = (int)((inputs[i][0] + 1) * 0.5 * WINDOW_WIDTH);
        int screenY = (int)((1 - (inputs[i][1] + 1) * 0.5) * WINDOW_HEIGHT);

        if (targets[i] == 1) {
            SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
        } else {
            SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
        }

        SDL_Rect rect = {screenX - 2, screenY - 2, 4, 4};
        SDL_RenderFillRect(renderer, &rect);
    }

    // Draw decision boundary
    double w0 = perceptron->weights[0];
    double w1 = perceptron->weights[1];
    double b = perceptron->bias;

    if (fabs(w1) > 1e-6) { // Avoid division by zero
        int x1 = 0, x2 = WINDOW_WIDTH;
        int y1 = (int)((1 - (-(b + w0 * -1) / w1 + 1) * 0.5) * WINDOW_HEIGHT);
        int y2 = (int)((1 - (-(b + w0 * 1) / w1 + 1) * 0.5) * WINDOW_HEIGHT);

        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
        SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
    }

    SDL_RenderPresent(renderer);
}

int main(int argc, char *argv[]) {
    const char *filename = "/home/marya/Documents/project/perceptron_dataset.csv";
    generateCSV(filename, NUM_SAMPLES);

    double inputs[NUM_SAMPLES][NUM_INPUTS];
    int targets[NUM_SAMPLES];

    if (!loadDataset(filename, inputs, targets, NUM_SAMPLES)) {
        return 1;
    }

    Perceptron perceptron;
    perceptron_init(&perceptron);

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("Perceptron Visualization", 500, 200, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        printf("SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
        SDL_DestroyWindow(window);
        printf("SDL_CreateRenderer Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    for (int epoch = 0; epoch < 5000; ++epoch) {
        int misclassified = 0;

        for (size_t i = 0; i < NUM_SAMPLES; ++i) {
            if (perceptron_classify(&perceptron, inputs[i]) != targets[i]) {
                perceptron_updateWeights(&perceptron, inputs[i], targets[i], 0.05);
                misclassified++;
            }
        }

        double accuracy = 1.0 - (double)misclassified / NUM_SAMPLES;
        printf("Epoch: %d | Accuracy: %.2f%% | Misclassified: %d\n", epoch + 1, accuracy * 100, misclassified);

        visualize(renderer, inputs, targets, NUM_SAMPLES, &perceptron);

        SDL_Delay(100);
    }

    printf("Training complete after 5000 epochs.\n");

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
