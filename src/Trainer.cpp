#include "Trainer.h"

Trainer::Trainer(Network& network) : net(network) {}

void Trainer::train(const std::vector<std::vector<double>> &inputs, 
                    const std::vector<std::vector<double>> &targets, 
                    size_t epochs, double learningRate,     
                    const std::function<double(double)> &activation, 
                    const std::function<double(double)> &activationDerivative,
                    const std::function<double(double,double)> &lossDerivative,
                    const std::function<double(const std::vector<double>&, const std::vector<double>&)> &lossFunction, 
                    size_t printEvery){

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            net.backpropagate(inputs[i], targets[i], activation, activationDerivative, learningRate,lossDerivative);
        }

        if (printEvery > 0 && epoch % printEvery == 0) {
            std::cout << "Epoch " << epoch << ":\n";
            evaluate(inputs, targets, activation,lossFunction,epoch);
            std::cout << "----------------------\n";
        }
    }
}

void Trainer::evaluate(const std::vector<std::vector<double>> &inputs, 
                       const std::vector<std::vector<double>> &targets, 
                       const std::function<double(double)> &activation,
                       const std::function<double(const std::vector<double>&, const std::vector<double>&)> &lossFunction,
                       size_t currentEpoch) const
{
    double totalLoss = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.feedForward(inputs[i], activation);
        totalLoss += lossFunction(output, targets[i]);
    }

    double avgLoss = totalLoss / inputs.size();
    std::cout << "Epoch " << currentEpoch << " - Avg Loss: " << std::fixed << std::setprecision(6) << avgLoss << "\n";
}

double Trainer::calculateRegressionAccuracy(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targets, const std::function<double(double)> &activation, double tolerance) const
{
    size_t correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.feedForward(inputs[i], activation);
        if (std::abs(output[0] - targets[i][0]) < tolerance) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / inputs.size();
}

double Trainer::calculateClassificationAccuracy(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targets, const std::function<double(double)> &activation) const
{
    size_t correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.feedForward(inputs[i], activation);

        int predicted = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        int actual = std::distance(targets[i].begin(), std::max_element(targets[i].begin(), targets[i].end()));

        if (predicted == actual)
            ++correct;
    }
    return static_cast<double>(correct) / inputs.size();
}

std::vector<double> Trainer::predictProbabilities(Network &net, const std::vector<double> &rawInput, const std::vector<double> &minVals, const std::vector<double> &maxVals)
{
    auto normalized = utils::normalize(rawInput, minVals, maxVals);
    auto output = net.feedForward(normalized, ActivationFunctions::sigmoid);
    auto probabilities = utils::softmax(output);
    return probabilities;
}

void Trainer::printClassProbabilities(const std::vector<double> &probabilities, const std::vector<std::string> &classLabels)
{
    std::cout << "Class probabilities:\n";
    for (size_t i = 0; i < probabilities.size(); ++i) {
        std::cout << "- " << classLabels[i] << ": "
                  << std::fixed << std::setprecision(2)
                  << probabilities[i] * 100 << "%\n";
    }
}

