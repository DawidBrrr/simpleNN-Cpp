#include "TrainerRegression.h"

TrainerRegression::TrainerRegression(Network& network,
                                     std::function<double(double)> activation,
                                     std::function<double(double)> activationDerivative,
                                     std::function<double(double, double)> lossDerivative,
                                     std::function<double(const std::vector<double>&, const std::vector<double>&)> lossFunction,
                                     double tolerance)
    : Trainer(network), activation(activation), activationDerivative(activationDerivative),
      lossDerivative(lossDerivative), lossFunction(lossFunction), tolerance(tolerance) {}

void TrainerRegression::train(const std::vector<std::vector<double>>& inputs,
                              const std::vector<std::vector<double>>& targets,
                              size_t epochs,
                              double learningRate,
                              size_t printEvery) {

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            net.backpropagate(inputs[i], targets[i], activation, activationDerivative, learningRate, lossDerivative);
        }

        if (printEvery > 0 && epoch % printEvery == 0) {
            std::cout << "Epoch " << epoch << ":\n";
            evaluate(inputs, targets, epoch);
            std::cout << "----------------------\n";
        }
    }
}

void TrainerRegression::evaluate(const std::vector<std::vector<double>>& inputs,
                                 const std::vector<std::vector<double>>& targets,
                                 size_t currentEpoch) const {
    double totalLoss = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.feedForward(inputs[i], activation);
        totalLoss += lossFunction(output, targets[i]);
    }
    double avgLoss = totalLoss / inputs.size();
    double accuracy = calculateAccuracy(inputs, targets);
    std::cout << "Epoch " << currentEpoch << " - Avg Loss: "
              << std::fixed << std::setprecision(6) << avgLoss << " - Accuracy: " 
              << std::fixed << std::setprecision(2) << accuracy << "\n";
}

double TrainerRegression::calculateAccuracy(const std::vector<std::vector<double>>& inputs,
                                            const std::vector<std::vector<double>>& targets) const {
    size_t correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.feedForward(inputs[i], activation);
        if (std::abs(output[0] - targets[i][0]) < tolerance) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / inputs.size();
}
