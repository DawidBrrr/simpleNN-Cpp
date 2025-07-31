#pragma once

#include <cmath>
#include <functional>

namespace ActivationFunctions{
    double sigmoid(double x);
    double sigmoidDerivative(double x);
    double relu(double x);
    double reluDerivative(double x);
    double tanh(double x);
}

namespace utils{
    std::vector<double> softmax(const std::vector<double>& values); 
    std::vector<double> normalize(const std::vector<double>& input, const std::vector<double>& minValues, const std::vector<double>& maxValues);
    std::pair<std::vector<double>, std::vector<double>> calculateMinMax(const std::vector<std::vector<double>>& inputs);  

}