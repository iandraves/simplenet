#include <stdlib.h>
#include "layer.h"

Layer new_layer(int activations, int weightRows, int weightColumns, int bias)
{
    Layer layer;

    layer.activations = create_matrix(activations, 1);
    layer.zValues = create_matrix(activations, 1);

    layer.weights = create_matrix(weightRows, weightColumns);
    layer.weightGrad = create_matrix(weightRows, weightColumns);

    layer.bias = create_matrix(bias, 1);
    layer.biasGrad = create_matrix(bias, 1);

    return layer;
}

Layer input_layer()
{
    Layer layer;

    layer.activations = create_matrix(INPUT_SIZE, 1);
    layer.zValues = create_null_matrix();

    layer.weights = create_null_matrix();
    layer.weightGrad = create_null_matrix();

    layer.bias = create_null_matrix();
    layer.biasGrad = create_null_matrix();

    return layer;
}