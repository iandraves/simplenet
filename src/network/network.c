#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "network.h"

/* --- INITIALIZATION --- */
void build_network(Network network)
{
    srand(time(0));
    network[0] = input_layer();
    // set_vals_to_zero(&network[0].activations);

    network[1] = new_layer(40, 40, 784, 40);
    set_vals_to_zero(&network[1].zValues);
    set_vals_to_zero(&network[1].activations);

    network[2] = new_layer(40, 40, 40, 40);
    set_vals_to_zero(&network[2].zValues);
    set_vals_to_zero(&network[2].activations);

    network[3] = new_layer(10, 10, 40, 10);
    set_vals_to_zero(&network[3].zValues);
    set_vals_to_zero(&network[3].activations);
}

/* --- FEED FORWARD --- */
void forward_layer(Layer *previous, Layer *forward)
{
    vector_dot_product(previous, forward); // Weights X Activations
    add(forward);
    activation_map(forward);
}

void final_forward(Layer *previous, Layer *final)
{
    vector_dot_product(previous, final);
    add(final);
    softmax_map(final);
}

void totalForward(Network network)
{
    forward_layer(&network[0], &network[1]);
    forward_layer(&network[1], &network[2]);
    final_forward(&network[2], &network[3]);
}

void forward(Network network, int trainNumber)
{
    int numberLabel = mount(network, trainNumber);

    totalForward(network);

    log_network(network, numberLabel);
}

/* --- COST --- */
float cross_entropy_loss(Network network, int label)
{
    Matrix output = network[NETWORK_SIZE - 1].activations;
    return -log(output.matrix[label][0]);
}

// float mseLoss(Network network, int label){
//     Matrix output = network[NETWORK_SIZE -1].activations;

//     float lossSum = 0.0;
//     for(int i = 0; i < output.rows; i++){
//         if(i == label){
//             lossSum += ( 1 - powf(output.matrix[i][0], 2.0) );
//         }else{
//             lossSum += powf(output.matrix[i][0], 2.0);
//         }
//     }

// }