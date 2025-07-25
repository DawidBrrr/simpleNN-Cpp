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
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.feedForward(inputs[i], ActivationFunctions::sigmoid);
        int predicted = output[0] >= 0.5 ? 1 : 0;
        int actual = static_cast<int>(targets[i][0]);
        if (predicted == actual) correct++;
    }

    double accuracy = static_cast<double>(correct) / inputs.size();
    std::cout << "XOR accuracy: " << accuracy << "\n";

    // Spodziewamy się 100% poprawnych odpowiedzi
    ASSERT_EQ(correct, 4);
}
