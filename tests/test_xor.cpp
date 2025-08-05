#include <gtest/gtest.h>
#include "Network.h"
#include "TrainerRegression.h"
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

    TrainerRegression trainer(net,
                                  ActivationFunctions::sigmoid,
                                  ActivationFunctions::sigmoidDerivative,
                                  LossFunctions::mseDerivative,
                                  LossFunctions::mse,0.15);
    trainer.train(inputs, targets, 10000, 0.5,1000);

    // Ewaluacja końcowa
    trainer.evaluate(inputs, targets,10000);

    double accuracy = trainer.calculateAccuracy(inputs, targets);

    std::cout << "Dokładność XOR (tolerancja 0.15): " << accuracy << "\n";
    EXPECT_GT(accuracy, 0.95);
}
