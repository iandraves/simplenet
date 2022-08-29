#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include "matrix.h"

/* --- INITIALIZATION --- */
Matrix create_null_matrix()
{
    Matrix matrix;

    matrix.rows = -1;
    matrix.columns = -1;

    return matrix;
}

Matrix create_matrix(int rows, int columns)
{
    Matrix matrix;

    // Set rows and columns
    matrix.rows = rows;
    matrix.columns = columns;

    // Allocate matrix array
    float **arr = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
        arr[i] = (float *)malloc(columns * sizeof(float));
    matrix.matrix = arr;

    // Initialize matrix with random values
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            matrix.matrix[i][j] = get_random_float(-1.0, 1.0);

    return matrix;
}

void set_vals_to_zero(Matrix *matrix)
{
    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->columns; j++)
            matrix->matrix[i][j] = 0.0;
}

void set_random(Matrix *matrix)
{
    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->columns; j++)
            matrix->matrix[i][j] = get_random_float(-1.0, 1.0);
}

/* --- OPERATIONS --- */
void add(Layer *layer)
{
    assert(layer->zValues.rows == layer->bias.rows);
    assert(layer->bias.rows == layer->activations.rows);
    assert(layer->zValues.columns == layer->bias.columns);
    assert(layer->bias.columns == layer->activations.columns);

    for (int i = 0; i < layer->zValues.rows; i++)
        for (int j = 0; j < layer->zValues.columns; j++)
            layer->zValues.matrix[i][j] = layer->zValues.matrix[i][j] + layer->bias.matrix[i][j];
}

void softmax_map(Layer *layer)
{
    assert(layer->zValues.rows == layer->activations.rows);
    assert(layer->zValues.columns == layer->activations.columns);
    assert(layer->zValues.columns == 1);

    float max = -INFINITY;
    for (int i = 0; i < layer->zValues.rows; i++)
        if (layer->zValues.matrix[i][0] > max)
            max = layer->zValues.matrix[i][0];

    float exponentSum = 0.0;
    for (int i = 0; i < layer->zValues.rows; i++)
    {
        float sumComponent = expf(layer->zValues.matrix[i][0] - max);
        exponentSum += sumComponent;
    }

    assert(exponentSum > 0.0);

    float offset = max + logf(exponentSum);
    for (int i = 0; i < layer->zValues.rows; i++)
        layer->activations.matrix[i][0] = expf(layer->zValues.matrix[i][0] - offset);

    // for(int i = 0; i < activations.rows; i++){
    //     printf("softmax %f \n", activations.matrix[1][i]);
    // }
}

void activation_map(Layer *layer)
{
    assert(layer->zValues.columns == 1);
    assert(layer->activations.columns == 1);

    for (int i = 0; i < layer->activations.rows; i++)
        layer->activations.matrix[i][0] = discrete_tanh(layer->zValues.matrix[i][0]);
}

void vector_dot_product(Layer *first, Layer *second)
{
    // printf("weights %d \n ", weights.columns);
    // printf("activations %d \n ", activations.rows);

    assert(second->weights.rows == second->zValues.rows);
    assert(second->weights.columns == first->activations.rows);

    for (int i = 0; i < second->weights.rows; i++)
    {
        // printf("Activations \n");
        float sum = 0.0;
        for (int j = 0; j < first->activations.rows; j++)
        {
            // printf(" ( %f", first -> activations.matrix[j][0]);
            // printf(", %f ) \n ", second -> weights.matrix[i][j]);
            sum += (first->activations.matrix[j][0] * second->weights.matrix[i][j]);
        }

        // printf("sum %f \n" , sum);

        second->zValues.matrix[i][0] = sum;
    }
}

// void multiply(Matrix * m1, Matrix * m2, Matrix * output)
// {
//     assert((m1->rows == m2->rows) && (m2->rows == output->rows));
//     assert((m1->columns == m2->columns) && (m2->columns == output->columns));

//     for (int i = 0; i < m1->rows; i++)
//     {
//         for (int j = 0; j < m1->columns; j++)
//         {
//             output->matrix[i][j] = m1->matrix[i][j] * m2->matrix[i][j];
//         }
//     }
// }

/* --- HELPERS --- */
float get_random_float(float min, float max)
{
    return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}

float discrete_tanh(float arg)
{
    if (arg < -1.8)
        return -1.0;
    else if (arg < -0.657)
        return ((0.3 * arg) - 0.46);
    else if (arg < 0.657)
        return arg;
    else if (arg < 1.8)
        return ((0.3 * arg) + 0.46);
    else
        return 1.0;
}

/* --- DEBUGGING --- */
void print_matrix(Matrix matrix)
{
    printf("{\n");
    for (int i = 0; i < matrix.rows; i++)
    {
        printf("    {");
        for (int j = 0; j < matrix.columns; j++)
        {
            if (j == matrix.columns - 1)
                printf("%f", matrix.matrix[i][j]);
            else
                printf("%f, ", matrix.matrix[i][j]);
        }

        if (i == matrix.rows - 1)
            printf("}\n");
        else
            printf("},\n");
    }
    printf("}\n");
}