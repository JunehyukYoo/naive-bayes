#include <iostream>

#include <core/data_model.h>
#include <fstream>

// TODO: You may want to change main's signature to take in argc and argv
int main() {
  std::string file_path_test = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/testtrainingimages.txt";
  std::string file_path_full = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/trainingimagesandlabels.txt";
  naivebayes::DataModel model;
  std::ifstream input_file(file_path_full);
  if (input_file.is_open()) {
    input_file >> model;
  } else {
    std::cerr << "error_message" << std::endl;
  }
  return 0;
}
