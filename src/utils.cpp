#include "utils.h"

double ActivationFunctions::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double ActivationFunctions::relu(double x)
{
    return x > 0 ? x : 0;
}

double ActivationFunctions::tanh(double x)
{
    return std::tanh(x);
}
