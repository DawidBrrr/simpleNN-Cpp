#include "TrainerClassification.h"
#include <random>
#include <algorithm>

/*
Trainer for classification tasks
@network: The neural network to train
@activation: The activation function to use
@activationDerivative: The derivative of the activation function
@lossDerivative: The derivative of the loss function
@lossFunction: The loss function to use
*/ 
TrainerClassification::TrainerClassification(Network& network,
                                             std::function<double(double)> activation,
                                             std::function<double(double)> activationDerivative,
                                             std::function<double(double, double)> lossDerivative,
                                             std::function<double(const std::vector<double>&, const std::vector<double>&)> lossFunction)
    : Trainer(network), activation(activation), activationDerivative(activationDerivative),
      lossDerivative(lossDerivative), lossFunction(lossFunction) {}

void TrainerClassification::train(const std::vector<std::vector<double>>& inputs,
                                  const std::vector<std::vector<double>>& targets,
                                  size_t epochs,
                                  double learningRate,
                                  size_t printEvery,
                                  size_t batchSize) {

    size_t numSamples = inputs.size();

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle data for better training
        std::vector<size_t> indices(numSamples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        // Process data in batches
        size_t numBatches = (numSamples + batchSize - 1) / batchSize; // Ceiling division
        for (size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
            size_t start = batchIdx * batchSize;
            size_t end = std::min(start + batchSize, numSamples);
            size_t currentBatchSize = end - start;

            std::vector<std::vector<double>> batchInputs;
            std::vector<std::vector<double>> batchTargets;

            for (size_t i = start; i < end; ++i) {
                batchInputs.push_back(inputs[indices[i]]);
                batchTargets.push_back(targets[indices[i]]);
            }

            if (currentBatchSize == 1) {
                // Single sample
                net.backpropagate(batchInputs[0], batchTargets[0], activation, activationDerivative, learningRate, lossDerivative);
            } else {
                // Batch
                net.backpropagateBatch(batchInputs, batchTargets, activation, activationDerivative, learningRate, lossDerivative);
            }

            // Display progress for each batch if printEvery > 0
            if (printEvery > 0) {
                std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs 
                          << " - Batch " << (batchIdx + 1) << "/" << numBatches 
                          << " (" << std::fixed << std::setprecision(1) 
                          << (static_cast<double>(batchIdx + 1) / numBatches * 100) << "%)" << std::flush;
            }
        }

        if (printEvery > 0) {
            std::cout << std::endl; // New line after epoch progress
        }

        if (printEvery > 0 && epoch % printEvery == 0) {
            std::cout << "Epoch " << epoch << ":\n";
            evaluate(inputs, targets, epoch);
            std::cout << "----------------------\n";
        }
    }
}

void TrainerClassification::evaluate(const std::vector<std::vector<double>>& inputs,
                                     const std::vector<std::vector<double>>& targets,
                                     size_t currentEpoch) const {
    double totalLoss = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.feedForward(inputs[i], activation);
        totalLoss += lossFunction(utils::softmax(output), targets[i]);
    }
    double avgLoss = totalLoss / inputs.size();
    double accuracy = calculateAccuracy(inputs, targets);
    std::cout << "Epoch " << currentEpoch << " - Avg Loss: "
              << std::fixed << std::setprecision(6) << avgLoss << " - Accuracy: " 
              << std::fixed << std::setprecision(2) << accuracy << "\n";
}

double TrainerClassification::calculateAccuracy(const std::vector<std::vector<double>>& inputs,
                                                const std::vector<std::vector<double>>& targets) const {
    size_t correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.feedForward(inputs[i], activation);
        auto probabilities = utils::softmax(output);

        int predicted = static_cast<int>(std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end())));
        int actual = static_cast<int>(std::distance(targets[i].begin(), std::max_element(targets[i].begin(), targets[i].end())));

        if (predicted == actual)
            ++correct;
    }
    return static_cast<double>(correct) / inputs.size();
}

std::vector<double> TrainerClassification::predictProbabilities(Network &net, const std::vector<double> &rawInput, const std::vector<double> &minVals, const std::vector<double> &maxVals)
{
    auto normalized = utils::normalize(rawInput, minVals, maxVals);
    auto output = net.feedForward(normalized, ActivationFunctions::sigmoid);
    auto probabilities = utils::softmax(output);
    return probabilities;
}

void TrainerClassification::printClassProbabilities(const std::vector<double> &probabilities, const std::vector<std::string> &classLabels)
{
    std::cout << "Class probabilities:\n";
    for (size_t i = 0; i < probabilities.size(); ++i) {
        std::cout << "- " << classLabels[i] << ": "
                  << std::fixed << std::setprecision(2)
                  << probabilities[i] * 100 << "%\n";
    }
}
