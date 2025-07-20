#include "Layer.h"

//Parametized Contructor
Layer::Layer(size_t neuronCount, size_t inputCountPerNeuron)
    : neurons(neuronCount), weights(neuronCount, std::vector<double>(inputCountPerNeuron, 0.0)) {}

//Copy Constructor
Layer::Layer(const Layer &other) : neurons(other.neurons), weights(other.weights) {}

//Move constructor
Layer::Layer(Layer &&other) noexcept : neurons(std::move(other.neurons)), weights(std::move(other.weights)) {}

//Copy assignment operator
Layer &Layer::operator=(const Layer &other)
{
    if(this != &other){
        neurons = other.neurons;
        weights = other.weights;
    }

    return *this;
}
//Move assignment operator
Layer &Layer::operator=(Layer &&other) noexcept
{
    if(this != &other){
        neurons = std::move(other.neurons);
        weights = std::move(other.weights);
    }
    return *this;
}

// Non-const version (allows modification)
std::vector<Neuron>& Layer::getNeurons() {
    return neurons;
}

// Const version (for read-only access)
const std::vector<Neuron>& Layer::getNeurons() const {
    return neurons;
}

const std::vector<std::vector<double>> &Layer::getWeights() const
{
    return weights;
}


void Layer::setWeights(const std::vector<std::vector<double>> &newWeights)
{
    //Validate the new weights structure matches expected dimensions
    if(newWeights.size() != neurons.size()){
        throw std::invalid_argument("Number of weight vectors (" + std::to_string(newWeights.size()) +
        ") doesn't match number of neurons (" + std::to_string(neurons.size()) + ")");
    }

    for(size_t i = 0; i < weights.size(); i++){
        if(newWeights[i].size() != weights[i].size()){
            throw std::invalid_argument(
                "Weight vector[" + std::to_string(i) + "] size (" + 
                std::to_string(newWeights[i].size()) + 
                ") doesn't match expected size (" + 
                std::to_string(weights[i].size()) + ")"
            );
        }
    }

    weights = newWeights; // Only assigned if validation passes

}

std::vector<double> Layer::getBiases() const
{
    std::vector<double> biases;
    biases.reserve(neurons.size());
    for(const auto& neuron : neurons){
        biases.push_back(neuron.getBias());
    }
    return biases;
}

void Layer::setBiases(const std::vector<double> &newBiases)
{
    if(newBiases.empty()){
        throw std::invalid_argument("Biases vector cannot be empty");
    }

    // Validate size matches number of neurons
    if (newBiases.size() != neurons.size()) {
        throw std::invalid_argument(
            "Biases count (" + std::to_string(newBiases.size()) + 
            ") must match neuron count (" + 
            std::to_string(neurons.size()) + ")"
        );
    }

    for(size_t i = 0; i < neurons.size();i++){
        neurons[i].setBias(newBiases[i]);
    }
}
/**
 * Performs forward propagation through the layer.
 * 
 * @param inputs Input values from previous layer (size must match weights[0].size())
 * @param activationFunction Nonlinear activation function to apply (e.g., sigmoid, ReLU)
 * @return Vector of activated outputs for each neuron in this layer
 * 
 * Computes: output[i] = activation(bias[i] + Σ(input[j] * weights[i][j]))
 */
std::vector<double> Layer::feedForward(const std::vector<double> &inputs, const std::function<double(double)> &activationFunction) const
{
    // Validate input dimensions match what we expect
    if (!weights.empty() && !weights[0].empty()) {
        assert(inputs.size() == weights[0].size() && 
               "Input size must match weight matrix column size");
    }

    std::vector<double> outputs;
    outputs.reserve(neurons.size()); // Pre-allocate space

    for(size_t i = 0; i < neurons.size(); i++){
        double weighted_sum = std::inner_product(
            inputs.begin(), inputs.end(),
            weights[i].begin(),
            neurons[i].getBias() //Start with the bias term
        );

        outputs.push_back(activationFunction(weighted_sum));
    }

    return outputs;

}
