#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include "../matrix/matrix.h"
#include "../layer/layer.h"
#include "../network/network.h"

void log_matrix(Matrix matrix, FILE *log);
void log_layer(Layer *layer, FILE *log, int layerNumber);
void log_network(Network network, int numberLabel);

#endif