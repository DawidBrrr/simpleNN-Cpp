#include "Network.h"
#include <numeric>

Network::Network(const std::vector<size_t>& topology){
    for(size_t i = 1; i < topology.size(); ++i){
        layers.emplace_back(topology[i],topology[i-1]);
    }
}

std::vector<double> Network::feedForward(const std::vector<double>& input, const std::function<double(double)>& activationFunction) const{
    
    std::vector<double> outputs = input;
    for(const auto& layer : layers){
        outputs = layer.feedForward(outputs, activationFunction);
    }

    return outputs;
}

void Network::backpropagate(const std::vector<double> &input, 
                            const std::vector<double> &target, 
                            const std::function<double(double)> &activationFunction, 
                            const std::function<double(double)> &activationDerivative, 
                            double learningRate,
                            const std::function<double(double,double)> &lossDerivative) {

    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> zs;

    // === 1. Forward Pass ===
    std::vector<double> a = input;
    activations.push_back(a);

    for (auto& layer : layers) {
        std::vector<double> z;
        std::vector<double> nextA;

        for (size_t i = 0; i < layer.getNeurons().size(); ++i) {
            double sum = std::inner_product(a.begin(), a.end(), layer.getWeights()[i].begin(), layer.getNeurons()[i].getBias());
            z.push_back(sum);
            nextA.push_back(activationFunction(sum));
        }

        zs.push_back(z);
        activations.push_back(nextA);
        a = nextA;
    }

    // === 2. Backward Pass ===

    std::vector<std::vector<double>> deltas(layers.size());

    // Output layer error
    size_t L = layers.size() - 1;
    deltas[L].resize(layers[L].getNeurons().size());

    for (size_t i = 0; i < layers[L].getNeurons().size(); ++i) {
        double z = zs[L][i];
        double a_val = activations[L + 1][i];
        deltas[L][i] = lossDerivative(a_val, target[i]) * activationDerivative(z);
    }

    // Hidden layers
    for (int l = L - 1; l >= 0; --l) {
        deltas[l].resize(layers[l].getNeurons().size());
        for (size_t i = 0; i < layers[l].getNeurons().size(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < layers[l + 1].getNeurons().size(); ++j) {
                sum += layers[l + 1].getWeights()[j][i] * deltas[l + 1][j];
            }
            deltas[l][i] = activationDerivative(zs[l][i]) * sum;
        }
    }

    // === 3. Update Weights and Biases ===
    for (size_t l = 0; l < layers.size(); ++l) {
        auto& weights = layers[l].getWeights();
        auto& neurons = layers[l].getNeurons();
        const auto& a_prev = activations[l];

        for (size_t i = 0; i < neurons.size(); ++i) {
            // Update bias
            double newBias = neurons[i].getBias() - learningRate * deltas[l][i];
            neurons[i].setBias(newBias);

            // Update weights
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] -= learningRate * deltas[l][i] * a_prev[j];
            }
        }
    }

}

void Network::backpropagateBatch(const std::vector<std::vector<double>>& inputs,
                                 const std::vector<std::vector<double>>& targets,
                                 const std::function<double(double)>& activationFunction,
                                 const std::function<double(double)>& activationDerivative,
                                 double learningRate,
                                 const std::function<double(double,double)> &lossDerivative) {

    size_t batchSize = inputs.size();
    if (batchSize == 0) return;

    // Initialize accumulated gradients for weights and biases
    std::vector<std::vector<std::vector<double>>> weightGradients(layers.size());
    std::vector<std::vector<double>> biasGradients(layers.size());

    for (size_t l = 0; l < layers.size(); ++l) {
        weightGradients[l].resize(layers[l].getNeurons().size());
        biasGradients[l].resize(layers[l].getNeurons().size());
        for (size_t i = 0; i < layers[l].getNeurons().size(); ++i) {
            weightGradients[l][i].resize(layers[l].getWeights()[i].size(), 0.0);
            biasGradients[l][i] = 0.0;
        }
    }

    // Process each example in the batch
    for (size_t b = 0; b < batchSize; ++b) {
        const auto& input = inputs[b];
        const auto& target = targets[b];

        std::vector<std::vector<double>> activations;
        std::vector<std::vector<double>> zs;

        // === 1. Forward Pass ===
        std::vector<double> a = input;
        activations.push_back(a);

        for (auto& layer : layers) {
            std::vector<double> z;
            std::vector<double> nextA;

            for (size_t i = 0; i < layer.getNeurons().size(); ++i) {
                double sum = std::inner_product(a.begin(), a.end(), layer.getWeights()[i].begin(), layer.getNeurons()[i].getBias());
                z.push_back(sum);
                nextA.push_back(activationFunction(sum));
            }

            zs.push_back(z);
            activations.push_back(nextA);
            a = nextA;
        }

        // === 2. Backward Pass ===
        std::vector<std::vector<double>> deltas(layers.size());

        // Output layer error
        size_t L = layers.size() - 1;
        deltas[L].resize(layers[L].getNeurons().size());

        for (size_t i = 0; i < layers[L].getNeurons().size(); ++i) {
            double z = zs[L][i];
            double a_val = activations[L + 1][i];
            deltas[L][i] = lossDerivative(a_val, target[i]) * activationDerivative(z);
        }

        // Hidden layers
        for (int l = L - 1; l >= 0; --l) {
            deltas[l].resize(layers[l].getNeurons().size());
            for (size_t i = 0; i < layers[l].getNeurons().size(); ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < layers[l + 1].getNeurons().size(); ++j) {
                    sum += layers[l + 1].getWeights()[j][i] * deltas[l + 1][j];
                }
                deltas[l][i] = activationDerivative(zs[l][i]) * sum;
            }
        }

        // === 3. Accumulate Gradients ===
        for (size_t l = 0; l < layers.size(); ++l) {
            const auto& a_prev = activations[l];

            for (size_t i = 0; i < layers[l].getNeurons().size(); ++i) {
                // Accumulate bias gradient
                biasGradients[l][i] += deltas[l][i];

                // Accumulate weight gradients
                for (size_t j = 0; j < layers[l].getWeights()[i].size(); ++j) {
                    weightGradients[l][i][j] += deltas[l][i] * a_prev[j];
                }
            }
        }
    }

    // === 4. Update Weights and Biases (average over batch) ===
    for (size_t l = 0; l < layers.size(); ++l) {
        auto& weights = layers[l].getWeights();
        auto& neurons = layers[l].getNeurons();

        for (size_t i = 0; i < neurons.size(); ++i) {
            // Update bias
            double avgBiasGrad = biasGradients[l][i] / batchSize;
            double newBias = neurons[i].getBias() - learningRate * avgBiasGrad;
            neurons[i].setBias(newBias);

            // Update weights
            for (size_t j = 0; j < weights[i].size(); ++j) {
                double avgWeightGrad = weightGradients[l][i][j] / batchSize;
                weights[i][j] -= learningRate * avgWeightGrad;
            }
        }
    }
}

void Network::initializeWeights(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0,1.0);

    for(auto& layer : layers){
        layer.initializeWeights([&]() {return dist(gen);});
    }

}

void Network::HeInitializeWeights(){
    std::random_device rd;
    std::mt19937 gen(rd());
    for (auto& layer : layers) {
        double stddev = std::sqrt(2.0 / layer.getNeurons().size());
        std::normal_distribution<double> dist(0.0, stddev);
        layer.initializeWeights([&]() { return dist(gen); });
    }
}

void Network::initializeBiases(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0,1.0);

    for(auto& layer : layers){
        layer.initializeBiases([&]() {return dist(gen);});
    }
}
