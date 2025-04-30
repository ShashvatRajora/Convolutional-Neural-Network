#include <stdio.h>
#include <stdlib.h>
#include "../include/cnn.h"
#include "../include/loss.h"
#include "../data/data_face80.h"

#define EPOCHS 50
#define LEARNING_RATE 0.001f

float normalized_images[480][32][32];

// Dummy labels: For now, assign image[i] → label = i % 10
unsigned char labels[480];

void normalize_dataset() {
    for (int i = 0; i < 480; i++) {
        for (int j = 0; j < 1024; j++) {
            normalized_images[i][j / 32][j % 32] = image[i][j] / 255.0f;
        }
    }
}

void prepare_labels() {
    for (int i = 0; i < 480; i++) {
        labels[i] = i % 10;
    }
}

int main() {
    normalize_dataset();
    prepare_labels();

    // Define layers
    ConvLayer1 conv1 = { .depth = 1, .size = FILTER_SIZE };
    ConvLayer2 conv2 = { .depth = CONV1_FILTERS, .size = FILTER_SIZE };
    FullyConnectedLayer fc = { .size = NUM_CLASSES };

    // Initialize weights and biases
    for (int f = 0; f < CONV1_FILTERS; f++) {
        for (int i = 0; i < FILTER_SIZE; i++)
            for (int j = 0; j < FILTER_SIZE; j++)
                conv1.filters[f][i][j] = 0.01f;
        conv1.biases[f] = 0.0f;
    }

    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int c = 0; c < CONV1_FILTERS; c++)
            for (int i = 0; i < FILTER_SIZE; i++)
                for (int j = 0; j < FILTER_SIZE; j++)
                    conv2.filters[f][c][i][j] = 0.01f;
        conv2.biases[f] = 0.0f;
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < 4096; j++)
            fc.weights[i][j] = 0.01f;
        fc.biases[i] = 0.0f;
    }

    // Buffers
    float conv1_output[32][32][CONV1_FILTERS] = {0};
    float conv2_output[32][32][CONV2_FILTERS] = {0};
    float pool_output[16][16][CONV2_FILTERS] = {0};
    float flat_output[4096] = {0};
    float d_flat_output[4096] = {0};
    float d_pool_output[16][16][CONV2_FILTERS] = {0};
    float fc_output[NUM_CLASSES] = {0};
    float softmax_output[NUM_CLASSES] = {0};
    float d_output[NUM_CLASSES] = {0};
    float d_conv2_output[32][32][CONV2_FILTERS] = {0};
    float d_conv1_output[32][32][CONV1_FILTERS] = {0};



    // 🚀 Training Loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;

        for (int img_idx = 0; img_idx < 480; img_idx++) {
            // Forward Pass
            conv_forward_1(normalized_images[img_idx], &conv1, conv1_output);
            conv_forward_2(conv1_output, &conv2, conv2_output);

            // ReLU
            for (int i = 0; i < 32; i++)
                for (int j = 0; j < 32; j++)
                    for (int f = 0; f < CONV2_FILTERS; f++)
                        if (conv2_output[i][j][f] < 0)
                            conv2_output[i][j][f] = 0;

            // Pool
            max_pool_forward(conv2_output, pool_output);

            // Flatten
            flatten(pool_output, flat_output);

            // FC Forward
            fc_forward(flat_output, &fc, fc_output);

            // Softmax
            softmax(fc_output, softmax_output);

            // Loss
            int label = labels[img_idx];
            float loss = cross_entropy_loss(softmax_output, label);
            total_loss += loss;

            // Gradient
            softmax_cross_entropy_derivative(softmax_output, label, d_output);

            // Backward Pass - Update FC Layer
            fc_backward(flat_output, &fc, d_output, d_flat_output, LEARNING_RATE);
            
            // Reshape back
            unflatten(d_flat_output, d_pool_output);
            
            // Backprop through Max Pool
            max_pool_backward(conv2_output, d_pool_output, d_conv2_output);

            // Backprop through Conv2
            conv2_backward(conv1_output, &conv2, d_conv2_output, d_conv1_output, LEARNING_RATE);

            // Backprop through Conv1
            conv1_backward(normalized_images[img_idx], &conv1, d_conv1_output, LEARNING_RATE);
        }

        // 🧠 Testing model on all 80 images after each epoch
        int correct = 0;
        for (int img_idx = 0; img_idx < 240; img_idx++) {
            conv_forward_1(normalized_images[img_idx], &conv1, conv1_output);
            conv_forward_2(conv1_output, &conv2, conv2_output);

            // ReLU
            for (int i = 0; i < 32; i++)
                for (int j = 0; j < 32; j++)
                    for (int f = 0; f < CONV2_FILTERS; f++)
                        if (conv2_output[i][j][f] < 0)
                            conv2_output[i][j][f] = 0;

            // Pool
            max_pool_forward(conv2_output, pool_output);

            // Flatten
            flatten(pool_output, flat_output);

            // FC Forward
            fc_forward(flat_output, &fc, fc_output);

            // Softmax
            softmax(fc_output, softmax_output);

            // Prediction
            int predicted = 0;
            float max_prob = softmax_output[0];
            for (int i = 1; i < NUM_CLASSES; i++) {
                if (softmax_output[i] > max_prob) {
                    max_prob = softmax_output[i];
                    predicted = i;
                }
            }

            if (predicted == labels[img_idx])
                correct++;
        }

        printf("Epoch %d: Average Loss = %.4f | Accuracy = %.2f%%\n", epoch + 1, total_loss / 480.0f, (correct / 480.0f) * 100.0f);
    }

    printf("\nTraining Finished!\n");
    save_model(&conv1, &conv2, &fc, "logs/trained_model.txt");


    // 🧪 Testing Predictions after Training
    printf("\nTesting Predictions after Training:\n");

    for (int img_idx = 0; img_idx < 10; img_idx++) {  // Test first 10 images
        // Forward Pass
        conv_forward_1(normalized_images[img_idx], &conv1, conv1_output);
        conv_forward_2(conv1_output, &conv2, conv2_output);

        // ReLU
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 32; j++)
                for (int f = 0; f < CONV2_FILTERS; f++)
                    if (conv2_output[i][j][f] < 0)
                        conv2_output[i][j][f] = 0;

        // Pool
        max_pool_forward(conv2_output, pool_output);

        // Flatten
        flatten(pool_output, flat_output);

        // FC Forward
        fc_forward(flat_output, &fc, fc_output);

        // Softmax
        softmax(fc_output, softmax_output);

        // Find predicted class (argmax)
        int predicted = 0;
        float max_prob = softmax_output[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (softmax_output[i] > max_prob) {
                max_prob = softmax_output[i];
                predicted = i;
            }
        }

        printf("Image %d - True Label: %d | Predicted: %d (Confidence: %.2f%%)\n", img_idx, labels[img_idx], predicted, max_prob * 100.0f);
    }

    return 0;
}
