#include <gtest/gtest.h>
#include "Network.h"
#include "TrainerClassification.h"
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
//TODO add a function making predictions / and improve NN
TEST(IrisNetworkTest, LearnsToClassify) {
    loadIrisDataset();

    Network net({4, 8, 3});
    net.initializeWeights();
    net.initializeBiases();
    TrainerClassification trainer(net,
                                  ActivationFunctions::sigmoid,
                                  ActivationFunctions::sigmoidDerivative,
                                  LossFunctions::crossEntropyDerivative,
                                  LossFunctions::crossEntropy);

    trainer.train(iris_inputs, iris_targets, 2000, 0.1,500);

    double accuracy = trainer.calculateAccuracy(iris_inputs, iris_targets);
    std::cout << "Iris classification accuracy: " << accuracy << "\n";

    ASSERT_GT(accuracy, 0.85); // minimum 85%
}