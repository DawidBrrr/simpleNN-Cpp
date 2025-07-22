#include "Trainer.h"

Trainer::Trainer(Network& network) : net(network) {}

void Trainer::train(const std::vector<std::vector<double>> &inputs, 
                    const std::vector<std::vector<double>> &targets, 
                    size_t epochs, double learningRate,     
                    const std::function<double(double)> &activation, 
                    const std::function<double(double)> &activationDerivative, 
                    size_t printEvery){

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            net.backpropagate(inputs[i], targets[i], activation, activationDerivative, learningRate);
        }

        if (printEvery > 0 && epoch % printEvery == 0) {
            std::cout << "Epoch " << epoch << ":\n";
            evaluate(inputs, targets, activation);
            std::cout << "----------------------\n";
        }
    }
}

void Trainer::evaluate(const std::vector<std::vector<double>> &inputs, 
                        const std::vector<std::vector<double>> &targets, 
                        const std::function<double(double)> &activation) const{

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.feedForward(inputs[i], activation);
        std::cout << "Input: [";
        for (double val : inputs[i]) {
            std::cout << val << " ";
        }
        std::cout << "] -> Output: "
                  << std::fixed << std::setprecision(4) << output[0]
                  << " (target: " << targets[i][0] << ")\n";
    }

}
