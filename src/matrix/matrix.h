#ifndef MATRIX_H
#define MATRIX_H

#include "../layer/layer.h"

typedef struct
{
    int rows;
    int columns;
    float **matrix;
} Matrix;

/* --- INITIALIZATION --- */
Matrix create_null_matrix();
Matrix create_matrix(int rows, int columns);
void set_vals_to_zero(Matrix *matrix);
void set_random(Matrix *matrix);

/* --- OPERATIONS --- */
void add(Layer *layer);
void softmax_map(Layer *layer);
void activation_map(Layer *layer);
void vector_dot_product(Layer *first, Layer *second);

/* --- HELPERS --- */
float get_random_float(float min, float max);
float discrete_tanh(float arg);

/* --- DEBUGGING --- */
void print_matrix(Matrix matrix);

#endif