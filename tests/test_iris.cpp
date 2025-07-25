#include <gtest/gtest.h>
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
        std::vector<double> target(3, 0.0);
        if (label == "Iris-setosa") target[0] = 1.0;
        else if (label == "Iris-versicolor") target[1] = 1.0;
        else if (label == "Iris-virginica") target[2] = 1.0;

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
    Trainer trainer(net);

    trainer.train(iris_inputs, iris_targets, 2000, 0.1,
                  ActivationFunctions::sigmoid,
                  ActivationFunctions::sigmoidDerivative,
                  500);

    int correct = 0;
    for (size_t i = 0; i < iris_inputs.size(); ++i) {
        auto output = net.feedForward(iris_inputs[i], ActivationFunctions::sigmoid);

        int predicted = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        int actual = std::distance(iris_targets[i].begin(), std::max_element(iris_targets[i].begin(), iris_targets[i].end()));

        if (predicted == actual)
            correct++;
    }

    double accuracy = (double)correct / iris_inputs.size();
    std::cout << "Iris classification accuracy: " << accuracy << "\n";

    ASSERT_GT(accuracy, 0.85); // above at least 85%
}