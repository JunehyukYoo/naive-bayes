#include <iostream>

#include <core/data_model.h>
#include <fstream>

int main() {
  std::string file_path_test = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/testtrainingimages.txt";
  std::string file_path_full = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/trainingimagesandlabels.txt";
  std::string file_path_save = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/savefile.txt";
  std::string file_path_empty = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/emptysavefile.txt";
  std::string file_path_empty_training_model = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/emptytrainingimages.txt";
  naivebayes::DataModel model(3);
  naivebayes::DataModel model1(3);
  
  //>> operator, build model
  std::ifstream input_file(file_path_test);
  if (input_file.is_open()) {
    input_file >> model;
  } else {
    std::cerr << "error message" << std::endl;
  }
  
  //<< operator, save file
  std::ofstream output_file(file_path_save);
  if (output_file.is_open()) {
    output_file << model;
  } else {
    std::cerr << "error message" << std::endl;
  }
  
  //>> operator, load save file
  std::ifstream input_file_1(file_path_save);
  if (input_file_1.is_open()) {
    input_file_1 >> model1;
  } else {
    std::cerr << "error message" << std::endl;
  }
  
  std::ifstream input_file_2(file_path_empty);
  if (input_file_2.is_open()) {
    try {
      input_file_2 >> model1;
    } catch (...) {}
  } else {
    std::cerr << "error message" << std::endl;
  }
  
  naivebayes::DataModel model2(3);
  std::ifstream input_file_3(file_path_empty_training_model);
  if (input_file_3.is_open()) {
    try {
      input_file_3 >> model2;
    } catch (...) {}
  } else {
    std::cerr << "error message" << std::endl;
  }
  return 0;
}
