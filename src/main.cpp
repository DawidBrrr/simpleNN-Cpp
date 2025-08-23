#include "Network.h"
#include "TrainerClassification.h"
#include "utils.h"
#include "mnist_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>


int main(){
    try{
        auto train = mnist::loadCSV("../data/mnist_train.csv",60000);
        auto test = mnist::loadCSV("../data/mnist_test.csv",10000);

        std::cout << "Train samples: " << train.inputs.size() << std::endl;
        std::cout << "Test samples: " << test.inputs.size() << std::endl;

        //Network: 784 -> 128 -> 64 -> 10
        Network network({784, 128, 64, 10});
        network.HeInitializeWeights();
        network.initializeBiases();

        TrainerClassification trainer(network,
                                      ActivationFunctions::sigmoid,
                                      ActivationFunctions::sigmoidDerivative,
                                      LossFunctions::crossEntropyDerivative,
                                      LossFunctions::crossEntropy);
        
        trainer.train(train.inputs,train.targets,10,0.1,1);
        
        double accuracy = trainer.calculateAccuracy(test.inputs, test.targets);

        std::cout << "MNIST Test Accuracy: " << accuracy << std::endl;

    } catch(const std::exception &e){
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    

    return 0;
}