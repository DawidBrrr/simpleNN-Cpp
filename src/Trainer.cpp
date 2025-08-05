#include "Trainer.h"

Trainer::Trainer(Network& network, size_t printEvery)
    : net(network), printEvery(printEvery) {}
