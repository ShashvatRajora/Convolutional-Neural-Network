#include <math.h>
#include "../include/loss.h"

float cross_entropy_loss(float predicted[], int label) {
    return -logf(predicted[label] + 1e-7f);
}

void softmax_cross_entropy_derivative(float predicted[], int label, float d_output[]) {
    for (int i = 0; i < NUM_CLASSES; i++) {
        d_output[i] = predicted[i];
    }
    d_output[label] -= 1.0f;
}
