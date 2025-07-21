#pragma once

#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>
#include <cassert>

#include "neuron.h"

class Layer {
private:
    std::vector<Neuron> neurons;
    std::vector<std::vector<double>> weights; // weights[i][j]: from neuron j in previous layer to neuron i in this layer
public:
    Layer(size_t neuronCount, size_t inputCountPerNeuron);
    Layer(const Layer &other);
    Layer(Layer &&other) noexcept;
    Layer& operator=(const Layer &other);
    Layer& operator=(Layer &&other) noexcept;
    ~Layer() = default;    

    //Setter Getters
    std::vector<Neuron>& getNeurons();
    const std::vector<Neuron>& getNeurons() const;

    const std::vector<std::vector<double>>& getWeights() const;
    void setWeights(const std::vector<std::vector<double>>& newWeights);
    
    std::vector<double> getBiases() const;
    void setBiases(const std::vector<double>& newBiases);

    //Reset
    void reset();

    //FeedForward function
    std::vector<double> feedForward(const std::vector<double> &inputs, const std::function<double(double)> &activationFunction) const;

    //Random Initializations
    void initializeWeights(std::function<double()> generator);
    void initializeBiases(std::function<double()> generator);


};