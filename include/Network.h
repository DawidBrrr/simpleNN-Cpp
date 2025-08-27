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
                   double learningRate,
                   const std::function<double(double,double)> &lossDerivative);

    void backpropagateBatch(const std::vector<std::vector<double>>& inputs,
                            const std::vector<std::vector<double>>& targets,
                            const std::function<double(double)>& activationFunction,
                            const std::function<double(double)>& activationDerivative,
                            double learningRate,
                            const std::function<double(double,double)> &lossDerivative);

    void initializeWeights();
    void HeInitializeWeights();
    void initializeBiases();
};