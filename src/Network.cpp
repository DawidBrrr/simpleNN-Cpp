#include "Network.h"
#include <random>

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
                            const std::function<double(double)> &activationDerivative, double learningRate){


                                
}

void Network::initializeWeights(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0,1.0);

    for(auto& layer : layers){
        layer.initializeWeights([&]() {return dist(gen);});
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
