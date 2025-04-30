#ifndef CNN_H
#define CNN_H

#define IMG_WIDTH 32
#define IMG_HEIGHT 32
#define CONV1_FILTERS 8
#define CONV2_FILTERS 16
#define FILTER_SIZE 3
#define POOL_SIZE 2
#define NUM_CLASSES 10

typedef struct {
    int depth;
    int size;
    float filters[CONV1_FILTERS][FILTER_SIZE][FILTER_SIZE];
    float biases[CONV1_FILTERS];
} ConvLayer1;

typedef struct {
    int depth;
    int size;
    float filters[CONV2_FILTERS][CONV1_FILTERS][FILTER_SIZE][FILTER_SIZE];
    float biases[CONV2_FILTERS];
} ConvLayer2;

typedef struct {
    int size;
    float weights[NUM_CLASSES][4096];
    float biases[NUM_CLASSES];
} FullyConnectedLayer;

// Function Prototypes
void conv_forward_1(float input[IMG_HEIGHT][IMG_WIDTH], ConvLayer1 *layer, float output[IMG_HEIGHT][IMG_WIDTH][CONV1_FILTERS]);
void conv_forward_2(float input[IMG_HEIGHT][IMG_WIDTH][CONV1_FILTERS], ConvLayer2 *layer, float output[IMG_HEIGHT][IMG_WIDTH][CONV2_FILTERS]);
void relu_forward(float input[], int len);
void max_pool_forward(float input[32][32][CONV2_FILTERS], float output[16][16][CONV2_FILTERS]);
void flatten(float input[16][16][CONV2_FILTERS], float output[4096]);
void unflatten(float input[4096], float output[16][16][CONV2_FILTERS]);
void fc_forward(float input[4096], FullyConnectedLayer *fc, float output[NUM_CLASSES]);
void fc_backward(float input[4096], FullyConnectedLayer *fc, float d_output[NUM_CLASSES], float d_input[4096], float learning_rate);
void softmax(float input[NUM_CLASSES], float output[NUM_CLASSES]);
void max_pool_backward(float input[32][32][CONV2_FILTERS], float d_pool[16][16][CONV2_FILTERS], float d_conv2[32][32][CONV2_FILTERS]);
void conv2_backward(float input[32][32][CONV1_FILTERS], ConvLayer2 *layer, float d_output[32][32][CONV2_FILTERS], float d_input[32][32][CONV1_FILTERS], float learning_rate);
void conv1_backward(float input[32][32], ConvLayer1 *layer, float d_output[32][32][CONV1_FILTERS], float learning_rate);
void save_model(const ConvLayer1 *conv1, const ConvLayer2 *conv2, const FullyConnectedLayer *fc, const char *filename);

#endif
