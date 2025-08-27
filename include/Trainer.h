#pragma once

#include "Network.h"
#include <vector>
#include <functional>
#include <string>

class Trainer {
protected:
    Network& net;
    size_t printEvery;

public:
    Trainer(Network& network, size_t printEvery = 1000);

    virtual void train(const std::vector<std::vector<double>>& inputs,
                       const std::vector<std::vector<double>>& targets,
                       size_t epochs,
                       double learningRate,
                       size_t printEvery,
                       size_t batchSize) = 0;

    virtual void evaluate(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& targets,
                          size_t currentEpoch) const = 0;

    virtual double calculateAccuracy(const std::vector<std::vector<double>>& inputs,
                                     const std::vector<std::vector<double>>& targets) const = 0;
};