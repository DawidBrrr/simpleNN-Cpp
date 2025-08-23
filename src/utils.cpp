#include "utils.h"

double ActivationFunctions::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double ActivationFunctions::sigmoidDerivative(double x)
{
    double s = sigmoid(x);
    return s * (1 - s);
}

double ActivationFunctions::relu(double x)
{
    return x > 0 ? x : 0;
}

double ActivationFunctions::reluDerivative(double x)
{
    return x > 0 ? 1.0 : 0.0;
}

double ActivationFunctions::tanh(double x)
{
    return std::tanh(x);
}

std::vector<double> utils::softmax(const std::vector<double>& values){
    std::vector<double> result(values.size());
    double maxVal = *std::max_element(values.begin(), values.end());
    double sum = 0.0;

    for(size_t i = 0; i < values.size(); i++){
        result[i] = std::exp(values[i] - maxVal);
        sum += result[i];
    }

    for(double & val : result){
        val /= sum;
    }

    return result;


}

std::vector<double> utils::normalize(const std::vector<double> &input, const std::vector<double> &minValues, const std::vector<double> &maxValues)
{
    std::vector<double> normalized(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        if(maxValues[i] - minValues[i] == 0){
            normalized[i] = 0.0; // Handle division by zero
        } else {
            normalized[i] = (input[i] - minValues[i]) / (maxValues[i] - minValues[i]);
        }
    }
    return normalized;
}

std::pair<std::vector<double>, std::vector<double>> utils::calculateMinMax(const std::vector<std::vector<double>> &inputs)
{
    if(inputs.empty() || inputs[0].empty()) {
        return {{}, {}};
    }

    size_t featureCount = inputs[0].size();
    std::vector<double> minValues(featureCount, std::numeric_limits<double>::max());
    std::vector<double> maxValues(featureCount, std::numeric_limits<double>::lowest());

    for(const auto& input : inputs) {
        for(size_t i = 0; i < featureCount; ++i){
            minValues[i] = std::min(minValues[i], input[i]);
            maxValues[i] = std::max(maxValues[i], input[i]);
        }
    }

    return {minValues, maxValues};
}

double LossFunctions::mse(const std::vector<double> &predicted, const std::vector<double> &target)
{
    if(predicted.size() != target.size()) {
        throw std::invalid_argument("Size mismatch in MSE Loss");
    }
    double sum = 0.0;
    for(size_t i = 0; i < predicted.size(); ++i){
        double diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}

double LossFunctions::crossEntropy(const std::vector<double> &predicted, const std::vector<double> &target)
{
    if(predicted.size() != target.size()) {
        throw std::invalid_argument("Size mismatch in Cross Entropy Loss");
    }

    double epsilon = 1e-12; // To avoid log(0)
    double sum = 0.0;
    for(size_t i = 0; i < predicted.size(); ++i){
        double p = std::max(std::min(predicted[i], 1.0 - epsilon),epsilon);
        sum += target[i] * std::log(p);
    }

    return -sum;
}

double LossFunctions::mseDerivative(double predicted, double target)
{
    return 2.0 * (predicted - target);
}

double LossFunctions::crossEntropyDerivative(double predicted, double target)
{
    return predicted-target; // zakłada softmax + Cross Entropy
}
