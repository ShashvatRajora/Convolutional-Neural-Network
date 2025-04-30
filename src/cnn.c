#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/cnn.h"

void conv_forward_1(float input[IMG_HEIGHT][IMG_WIDTH], ConvLayer1 *layer, float output[IMG_HEIGHT][IMG_WIDTH][CONV1_FILTERS]) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
        for (int i = 1; i < IMG_HEIGHT - 1; i++) {
            for (int j = 1; j < IMG_WIDTH - 1; j++) {
                float sum = 0.0;
                for (int ki = 0; ki < FILTER_SIZE; ki++) {
                    for (int kj = 0; kj < FILTER_SIZE; kj++) {
                        int x = i + ki - 1;
                        int y = j + kj - 1;
                        sum += input[x][y] * layer->filters[f][ki][kj];
                    }
                }
                output[i][j][f] = sum + layer->biases[f];
            }
        }
    }
}

void conv_forward_2(float input[IMG_HEIGHT][IMG_WIDTH][CONV1_FILTERS], ConvLayer2 *layer, float output[IMG_HEIGHT][IMG_WIDTH][CONV2_FILTERS]) {
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 1; i < IMG_HEIGHT - 1; i++) {
            for (int j = 1; j < IMG_WIDTH - 1; j++) {
                float sum = 0.0;
                for (int c = 0; c < CONV1_FILTERS; c++) {
                    for (int ki = 0; ki < FILTER_SIZE; ki++) {
                        for (int kj = 0; kj < FILTER_SIZE; kj++) {
                            int x = i + ki - 1;
                            int y = j + kj - 1;
                            sum += input[x][y][c] * layer->filters[f][c][ki][kj];
                        }
                    }
                }
                output[i][j][f] = sum + layer->biases[f];
            }
        }
    }
}

void relu_forward(float input[], int len) {
    for (int i = 0; i < len; i++) {
        if (input[i] < 0) input[i] = 0;
    }
}

void max_pool_forward(float input[32][32][CONV2_FILTERS], float output[16][16][CONV2_FILTERS]) {
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 0; i < 32; i += 2) {
            for (int j = 0; j < 32; j += 2) {
                float max_val = input[i][j][f];
                for (int m = 0; m < 2; m++) {
                    for (int n = 0; n < 2; n++) {
                        int x = i + m;
                        int y = j + n;
                        if (input[x][y][f] > max_val) {
                            max_val = input[x][y][f];
                        }
                    }
                }
                output[i / 2][j / 2][f] = max_val;
            }
        }
    }
}

void flatten(float input[16][16][CONV2_FILTERS], float output[4096]) {
    int idx = 0;
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                output[idx++] = input[i][j][f];
            }
        }
    }
}

void unflatten(float input[4096], float output[16][16][CONV2_FILTERS]) {
    int idx = 0;
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                output[i][j][f] = input[idx++];
            }
        }
    }
}


void fc_forward(float input[4096], FullyConnectedLayer *fc, float output[NUM_CLASSES]) {
    for (int i = 0; i < NUM_CLASSES; i++) {
        float sum = fc->biases[i];
        for (int j = 0; j < 4096; j++) {
            sum += fc->weights[i][j] * input[j];
        }
        output[i] = sum;
    }
}

void softmax(float input[NUM_CLASSES], float output[NUM_CLASSES]) {
    float max_val = input[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
        output[i] /= sum;
    }
}

void fc_backward(float input[4096], FullyConnectedLayer *fc, float d_output[NUM_CLASSES], float d_input[4096], float learning_rate) {
    // Initialize d_input to zero
    for (int j = 0; j < 4096; j++) {
        d_input[j] = 0.0f;
    }

    // Update weights and biases, compute d_input
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < 4096; j++) {
            d_input[j] += fc->weights[i][j] * d_output[i]; // Backprop to input
            fc->weights[i][j] -= learning_rate * d_output[i] * input[j]; // Update weights
        }
        fc->biases[i] -= learning_rate * d_output[i]; // Update biases
    }
}

void max_pool_backward(float input[32][32][CONV2_FILTERS], float d_pool[16][16][CONV2_FILTERS], float d_conv2[32][32][CONV2_FILTERS]) {
    // Initialize gradient of conv2 output to zero
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                d_conv2[i][j][f] = 0.0f;
            }
        }
    }

    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                // Find max position again
                int max_i = i * 2;
                int max_j = j * 2;
                float max_val = input[max_i][max_j][f];

                for (int m = 0; m < 2; m++) {
                    for (int n = 0; n < 2; n++) {
                        int x = i * 2 + m;
                        int y = j * 2 + n;
                        if (input[x][y][f] > max_val) {
                            max_val = input[x][y][f];
                            max_i = x;
                            max_j = y;
                        }
                    }
                }

                // Pass gradient only to max position
                d_conv2[max_i][max_j][f] = d_pool[i][j][f];
            }
        }
    }
}

void conv2_backward(float input[32][32][CONV1_FILTERS], ConvLayer2 *layer, float d_output[32][32][CONV2_FILTERS], float d_input[32][32][CONV1_FILTERS], float learning_rate) {
    // Initialize d_input to zero
    for (int c = 0; c < CONV1_FILTERS; c++) {
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                d_input[i][j][c] = 0.0f;
            }
        }
    }

    // For each filter
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 1; i < 31; i++) {  // valid area (ignore padding border)
            for (int j = 1; j < 31; j++) {
                for (int c = 0; c < CONV1_FILTERS; c++) {
                    for (int ki = 0; ki < FILTER_SIZE; ki++) {
                        for (int kj = 0; kj < FILTER_SIZE; kj++) {
                            int x = i + ki - 1;
                            int y = j + kj - 1;
                            layer->filters[f][c][ki][kj] -= learning_rate * d_output[i][j][f] * input[x][y][c];
                            d_input[x][y][c] += layer->filters[f][c][ki][kj] * d_output[i][j][f];
                        }
                    }
                }
                // Update bias
                layer->biases[f] -= learning_rate * d_output[i][j][f];
            }
        }
    }
}

void conv1_backward(float input[32][32], ConvLayer1 *layer, float d_output[32][32][CONV1_FILTERS], float learning_rate) {
    // For each filter
    for (int f = 0; f < CONV1_FILTERS; f++) {
        for (int i = 1; i < 31; i++) {  // valid area (ignore padding border)
            for (int j = 1; j < 31; j++) {
                for (int ki = 0; ki < FILTER_SIZE; ki++) {
                    for (int kj = 0; kj < FILTER_SIZE; kj++) {
                        int x = i + ki - 1;
                        int y = j + kj - 1;
                        layer->filters[f][ki][kj] -= learning_rate * d_output[i][j][f] * input[x][y];
                    }
                }
                // Update bias
                layer->biases[f] -= learning_rate * d_output[i][j][f];
            }
        }
    }
}

void save_model(const ConvLayer1 *conv1, const ConvLayer2 *conv2, const FullyConnectedLayer *fc, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        printf("Error opening file for saving model.\n");
        return;
    }

    // Save Conv1 filters
    for (int f_idx = 0; f_idx < CONV1_FILTERS; f_idx++) {
        for (int i = 0; i < FILTER_SIZE; i++) {
            for (int j = 0; j < FILTER_SIZE; j++) {
                fprintf(f, "%f ", conv1->filters[f_idx][i][j]);
            }
        }
        fprintf(f, "\n");
    }

    // Save Conv1 biases
    for (int f_idx = 0; f_idx < CONV1_FILTERS; f_idx++) {
        fprintf(f, "%f ", conv1->biases[f_idx]);
    }
    fprintf(f, "\n");

    // Save Conv2 filters
    for (int f_idx = 0; f_idx < CONV2_FILTERS; f_idx++) {
        for (int c = 0; c < CONV1_FILTERS; c++) {
            for (int i = 0; i < FILTER_SIZE; i++) {
                for (int j = 0; j < FILTER_SIZE; j++) {
                    fprintf(f, "%f ", conv2->filters[f_idx][c][i][j]);
                }
            }
        }
        fprintf(f, "\n");
    }

    // Save Conv2 biases
    for (int f_idx = 0; f_idx < CONV2_FILTERS; f_idx++) {
        fprintf(f, "%f ", conv2->biases[f_idx]);
    }
    fprintf(f, "\n");

    // Save Fully Connected weights
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < 4096; j++) {
            fprintf(f, "%f ", fc->weights[i][j]);
        }
        fprintf(f, "\n");
    }

    // Save Fully Connected biases
    for (int i = 0; i < NUM_CLASSES; i++) {
        fprintf(f, "%f ", fc->biases[i]);
    }
    fprintf(f, "\n");

    fclose(f);
    printf("Model saved successfully to %s\n", filename);
}