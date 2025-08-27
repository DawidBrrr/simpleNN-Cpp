#include "TrainerRegression.h"
#include <random>
#include <algorithm>

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
