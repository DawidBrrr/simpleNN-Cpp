#pragma once

#include "Trainer.h"
#include "utils.h"
#include <iostream>
#include <iomanip>

class TrainerClassification : public Trainer {
private:
    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;
    std::function<double(double, double)> lossDerivative;
    std::function<double(const std::vector<double>&, const std::vector<double>&)> lossFunction;

public:
    TrainerClassification(Network& network,
                          std::function<double(double)> activation,
                          std::function<double(double)> activationDerivative,
                          std::function<double(double, double)> lossDerivative,
                          std::function<double(const std::vector<double>&, const std::vector<double>&)> lossFunction);

    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               size_t epochs,
               double learningRate,
               size_t printEvery = 1000) override;

    void evaluate(const std::vector<std::vector<double>>& inputs,
                  const std::vector<std::vector<double>>& targets,
                  size_t currentEpoch) const override;

    double calculateAccuracy(const std::vector<std::vector<double>>& inputs,
                             const std::vector<std::vector<double>>& targets) const override;


    std::vector<double> predictProbabilities(Network& net,
                                                const std::vector<double>& rawInput,
                                                const std::vector<double>& minVals,
                                                const std::vector<double>& maxVals);
    
    void printClassProbabilities(const std::vector<double>& probabilities,
                                const std::vector<std::string>& classLabels);                             
};

