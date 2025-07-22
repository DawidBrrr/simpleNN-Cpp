#include <iostream>
#include "Network.h"
#include "Trainer.h"
#include "utils.h"

int main(){
    
    //Problem XOR

    std::vector<std::vector<double>> inputs = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> targets = {
    {0}, {1}, {1}, {0}
    };

    Network net({2, 3, 1}); // 2 wejścia, 1 warstwa ukryta z 3 neuronami, 1 wyjście
    net.initializeWeights();
    net.initializeBiases();

    Trainer trainer(net);
    trainer.train(inputs, targets, 10000,0.5,ActivationFunctions::sigmoid,ActivationFunctions::sigmoidDerivative,1000);

    std::cout << "\n=== Final Evaluation ===\n";
    trainer.evaluate(inputs, targets, ActivationFunctions::sigmoid);
    



    return 0;
}