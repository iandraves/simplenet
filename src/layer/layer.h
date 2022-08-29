#ifndef LAYER_H
#define LAYER_H

#include "../matrix/matrix.h"

#define INPUT_SIZE 784

typedef struct
{
    Matrix activations;
    Matrix zValues;

    Matrix weights;
    Matrix weightGrad;

    Matrix bias;
    Matrix biasGrad;

} Layer;

Layer new_layer(int activations, int weightRows, int weightColumns, int bias);
Layer input_layer();

#endif