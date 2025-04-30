#ifndef LOSS_H
#define LOSS_H

#define NUM_CLASSES 10

float cross_entropy_loss(float predicted[], int label);
void softmax_cross_entropy_derivative(float predicted[], int label, float d_output[]);

#endif
