#include <iostream>

#include <core/data_model.h>
#include <fstream>

int main() {
  std::string file_path_test = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/smallsettrainingimages.txt";
  std::string file_path_full = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/trainingimagesandlabels.txt";
  std::string file_path_save = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/savefile.txt";
  std::string file_path_empty = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/emptysavefile.txt";
  std::string file_path_empty_training_model = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/emptytrainingimages.txt";
  std::string file_path_full_testing = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/testimagesandlabels.txt";

  naivebayes::DataModel model_final;
  
  
  //28 image size, testing accuracy
  std::ifstream input_file_large(file_path_full);
  if (input_file_large.is_open()) {
    input_file_large >> model_final;
  } else {
    std::cerr << "error message" << std::endl;
  }
  model_final.CalculateModelAccuracy(file_path_full_testing);
  return 0;
}
