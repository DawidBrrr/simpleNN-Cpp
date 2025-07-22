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