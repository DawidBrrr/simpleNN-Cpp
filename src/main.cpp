#include <iostream>
#include "neuron.h"

int main(){
    Neuron neuron(0.5);
    neuron.setValue(10);
    std::cout << "Neuron value: " << neuron.getValue() << std::endl;
    std::cout << "Neuron bias: " << neuron.getBias() << std::endl;

    



    return 0;
}