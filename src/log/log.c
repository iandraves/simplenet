#include "log.h"

void log_matrix(Matrix matrix, FILE *log)
{
    fprintf(log, "{\n");
    for (int i = 0; i < matrix.rows; i++)
    {
        fprintf(log, "    {");
        for (int j = 0; j < matrix.columns; j++)
        {
            if (j == matrix.columns - 1)
                fprintf(log, "%f", matrix.matrix[i][j]);
            else
                fprintf(log, "%f, ", matrix.matrix[i][j]);
        }

        if (i == matrix.rows - 1)
            fprintf(log, "}\n");
        else
            fprintf(log, "},\n");
    }
    fprintf(log, "}\n");
}

void log_layer(Layer *layer, FILE *log, int layerNumber)
{
    fprintf(log, "---------------------LAYER %d ----------------- \n", layerNumber);

    fprintf(log, "weights %d\n", layerNumber);
    log_matrix(layer->weights, log);

    fprintf(log, "bias %d\n", layerNumber);
    log_matrix(layer->bias, log);

    fprintf(log, "zvalues %d\n", layerNumber);
    log_matrix(layer->zValues, log);

    fprintf(log, "activation %d\n", layerNumber);
    log_matrix(layer->activations, log);
}

void log_network(Network network, int numberLabel)
{
    FILE *log;
    log = fopen("log.txt", "w");
    log_layer(&network[0], log, 0);
    log_layer(&network[1], log, 1);
    log_layer(&network[2], log, 2);
    log_layer(&network[3], log, 3);
    fprintf(log, "Cross entropy loss: %f", cross_entropy_loss(network, numberLabel));

    fclose(log);
}