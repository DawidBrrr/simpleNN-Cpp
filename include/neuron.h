#pragma once

#include <vector>

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

    int getValue() const;
    void setValue(int v);
    double getBias() const;
    void setBias(double b);

    void reset();



};