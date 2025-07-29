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
