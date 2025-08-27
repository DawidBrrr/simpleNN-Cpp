#pragma once

#include <vector>
#include <functional>
#include <utility>

class Neuron {
private:
    int value;
    double bias;

public:

    Neuron();
    Neuron(double bias);
    Neuron(const Neuron &other);
    Neuron(Neuron &&other) noexcept;
    Neuron& operator=(const Neuron &other);
    Neuron& operator=(Neuron &&other) noexcept;
    ~Neuron() = default;

    int getValue() const;
    void setValue(int v);
    double getBias() const;
    void setBias(double b);

    void reset();

    double activate(const std::function<double(double)> &activationFunction) const;



};