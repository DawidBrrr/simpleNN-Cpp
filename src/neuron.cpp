#include "neuron.h"

// Default constructor
Neuron::Neuron() : value(0), bias(0.0) {}

// Parameterized constructor
Neuron::Neuron(double bias) : value(0), bias(bias) {}

// Copy constructor
Neuron::Neuron(const Neuron &other) : value(other.value), bias(other.bias) {}

// Move constructor
Neuron::Neuron(Neuron &&other) noexcept 
    : value(std::exchange(other.value, 0)),  // Transfers value and sets other.value = 0
      bias(std::exchange(other.bias, 0.0))   // Transfers bias and sets other.bias = 0.0
{}

// Copy assignment operator
Neuron& Neuron::operator=(const Neuron &other) {
    if (this != &other) {
        value = other.value;
        bias = other.bias;
    }
    return *this;
}

// Move assignment operator
Neuron& Neuron::operator=(Neuron &&other) noexcept {
    if (this != &other) {
        value = other.value;
        bias = other.bias;
        other.value = 0;
        other.bias = 0.0;
    }
    return *this;
}

//Setter and getter methods
int Neuron::getValue() const {
    return value;
}

void Neuron::setValue(int v) {
    value = v;
}

double Neuron::getBias() const {
    return bias;
}

void Neuron::setBias(double b) {
    bias = b;
}
//Functions 
void Neuron::reset() {
    value = 0;
    bias = 0.0;
}

double Neuron::activate(const std::function<double(double)> &activationFunction) const {
    return activationFunction(static_cast<double>(value) + bias);
}