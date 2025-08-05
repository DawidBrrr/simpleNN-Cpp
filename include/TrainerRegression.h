#pragma once

#include "Trainer.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <iomanip>

class TrainerRegression : public Trainer {
private:
    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;
    std::function<double(double, double)> lossDerivative;
    std::function<double(const std::vector<double>&, const std::vector<double>&)> lossFunction;
    double tolerance;

public:
    TrainerRegression(Network& network,
                      std::function<double(double)> activation,
                      std::function<double(double)> activationDerivative,
                      std::function<double(double, double)> lossDerivative,
                      std::function<double(const std::vector<double>&, const std::vector<double>&)> lossFunction,
                      double tolerance = 0.1);


    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               size_t epochs,
               double learningRate,
               size_t printEvery = 1000) override;

    void evaluate(const std::vector<std::vector<double>>& inputs,
                  const std::vector<std::vector<double>>& targets,
                  size_t currentEpoch) const override;

    double calculateAccuracy(const std::vector<std::vector<double>> & inputs,
                           const std::vector<std::vector<double>> & targets) const override;
};