#include <stdio.h>
#include "./network/network.h"

int main(void)
{
    Network network;

    build_network(network);

    forward(network, 40);

    return 0;
}