#include <gtest/gtest.h>
#include "Network.h"
#include "Trainer.h"
#include "utils.h"

TEST(XORNetworkTest, LearnsXOR) {
    // Dane XOR
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> targets = {
        {0}, {1}, {1}, {0}
    };

    // Tworzenie sieci: 2 wejścia, 3 neurony w warstwie ukrytej, 1 wyjście
    Network net({2, 3, 1});
    net.initializeWeights();
    net.initializeBiases();

    Trainer trainer(net);
    trainer.train(inputs, targets, 10000, 0.5,
                  ActivationFunctions::sigmoid,
                  ActivationFunctions::sigmoidDerivative,
                  1000);

    // Ewaluacja końcowa
    trainer.evaluate(inputs, targets, ActivationFunctions::sigmoid);

    double accuracy = trainer.calculateRegressionAccuracy(
        inputs, targets, ActivationFunctions::sigmoid, 0.15);

    std::cout << "Dokładność XOR (tolerancja 0.15): " << accuracy << "\n";
    EXPECT_GT(accuracy, 0.95);
}
