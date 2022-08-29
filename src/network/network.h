#ifndef NETWORK_H
#define NETWORK_H

#include "../layer/layer.h"

#define NETWORK_SIZE 4

typedef Layer Network[NETWORK_SIZE];

/* --- INITIALIZATION --- */
void build_network(Network network);

/* --- FEED FORWARD --- */
void forward_layer(Layer *previous, Layer *forward);
void final_forward(Layer *previous, Layer *final);
void totalForward(Network network);
void forward(Network network, int trainNumber);

/* --- COST --- */
float cross_entropy_loss(Network network, int label);

#endif