#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mnist {
    struct Dataset {
        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> targets;
    };

    //Loading data from CSV
    inline Dataset loadCSV(const std::string& filepath,size_t limit = 0){
        Dataset dataset;
        std::ifstream file(filepath);
        if(!file.is_open()){
            throw std::runtime_error("Could not open file: " + filepath);
        }

        std::string line;
        size_t rowCount = 0;

        // Skip first row if it contains headers
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::getline(ss, cell, ',');
            try {
                std::stoi(cell); // Check if the first row contains a number
                file.seekg(0);   // If so, go back to the beginning of the file
            } catch (const std::invalid_argument&) {
                // If not, skip the first row
            }
        }


        while(std::getline(file,line)){
            std::stringstream ss(line);
            std::string cell;

            //First column aka Label
            std::getline(ss,cell,',');
            int label = std::stoi(cell);

            //Inputs
            std::vector<double> input;
            while(std::getline(ss,cell,',')){
                double val = std::stod(cell)/255.0; // Normalize
                input.push_back(val);
            }
            if(input.size() != 784){
                throw std::runtime_error("Invalid row size in MNIST CSV");
            }

            //One-hot encode target
            std::vector<double> target(10, 0.0);
            target[label] = 1.0;

            dataset.inputs.push_back(std::move(input));
            dataset.targets.push_back(std::move(target));
            rowCount++;
            if(limit > 0 && rowCount >= limit){
                break;
            }
        }

        return dataset;
    }
}