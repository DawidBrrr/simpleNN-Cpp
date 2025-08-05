#include "Network.h"
#include "Trainer.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

std::vector<std::vector<double>> iris_inputs;
std::vector<std::vector<double>> iris_targets;

void loadIrisDataset() {
    std::ifstream file("../data/iris.csv");
    std::string line;

    //Skip first line 
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;

        std::vector<double> input(4);
        for (int i = 0; i < 4; ++i) {
            std::getline(ss, val, ',');
            input[i] = std::stod(val);
        }

        std::string label;
        std::getline(ss, label, ',');
        label.erase(std::remove(label.begin(), label.end(), '\"'), label.end());
        std::vector<double> target(3, 0.0);
        if (label == "Setosa") target[0] = 1.0;
        else if (label == "Versicolor") target[1] = 1.0;
        else if (label == "Virginica") target[2] = 1.0;

        iris_inputs.push_back(input);
        iris_targets.push_back(target);
    }
}


int main(){
    
    //Problem Iris Classification
    loadIrisDataset();
    auto [minVals, maxVals] = utils::calculateMinMax(iris_inputs);
    for (auto& input : iris_inputs) {
        input = utils::normalize(input, minVals, maxVals);
    }

    Network net({4, 8, 3});
    net.initializeWeights();
    net.initializeBiases();
    Trainer trainer(net);

    trainer.train(iris_inputs, iris_targets, 2000, 0.1,
                  ActivationFunctions::sigmoid,
                  ActivationFunctions::sigmoidDerivative,
                  LossFunctions::mseDerivative,
                  LossFunctions::mse,
                  500);

    double accuracy = trainer.calculateClassificationAccuracy(iris_inputs, iris_targets, ActivationFunctions::sigmoid);
    std::cout << "Iris classification accuracy: " << accuracy << "\n";

    //Sample test
    std::vector<double> sample = {5.4,3.9,1.7,0.4};
    std::vector<std::string> classLabels = {"Setosa", "Versicolor", "Virginica"};

    auto probs = trainer.predictProbabilities(net, sample, minVals, maxVals);
    trainer.printClassProbabilities(probs, classLabels);


    return 0;
}