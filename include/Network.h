#pragma once

#include "Layer.h"
#include <random>
#include <vector>
#include <functional>

class Network{
private:
    std::vector<Layer> layers;

public:
    Network(const std::vector<size_t>& topology);

    std::vector<double> feedForward(const std::vector<double>& input, const std::function<double(double)>& activationFunction) const;

    void backpropagate(const std::vector<double>& input,
                   const std::vector<double>& target,
                   const std::function<double(double)>& activationFunction,
                   const std::function<double(double)>& activationDerivative,
                   double learningRate);

    void initializeWeights();
    void initializeBiases();
};