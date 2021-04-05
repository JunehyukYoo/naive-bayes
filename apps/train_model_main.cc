#include <iostream>

#include <core/data_model.h>
#include <fstream>

// TODO: You may want to change main's signature to take in argc and argv
int main() {
  // TODO: Replace this with code that reads the training data, trains a model,
  // and saves the trained model to a file.
  std::string file_path = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/testtrainingimages.txt";
  naivebayes::DataModel model(3);
  std::ifstream input_file(file_path);
  if (input_file.is_open()) {
    input_file >> model;
  } else {
    std::cerr << "error_message" << std::endl;
  }
  return 0;
}
