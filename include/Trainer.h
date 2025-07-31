#pragma once

#include "Network.h"
#include "utils.h"
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>

class Trainer{
private:
    Network& net;
public:
    Trainer(Network& network);

    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               size_t epochs,
               double learningRate,
               const std::function<double(double)>& activation,
               const std::function<double(double)>& activationDerivative,
               size_t printEvery = 1000);

    void evaluate(const std::vector<std::vector<double>>& inputs,
                  const std::vector<std::vector<double>>& targets,
                  const std::function<double(double)>& activation) const;

    double Trainer::calculateRegressionAccuracy(
                                                const std::vector<std::vector<double>>& inputs,
                                                const std::vector<std::vector<double>>& targets,
                                                const std::function<double(double)>& activation,
                                                double tolerance) const;

    double Trainer::calculateClassificationAccuracy(
                                                    const std::vector<std::vector<double>>& inputs,
                                                    const std::vector<std::vector<double>>& targets,
                                                    const std::function<double(double)>& activation) const;     

    std::vector<double> predictProbabilities(Network& net,
                                                const std::vector<double>& rawInput,
                                                const std::vector<double>& minVals,
                                                const std::vector<double>& maxVals);
    
    void printClassProbabilities(const std::vector<double>& probabilities,
                                const std::vector<std::string>& classLabels);                                                

};