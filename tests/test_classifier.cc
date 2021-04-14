#include <catch2/catch.hpp>

#include <fstream>
#include <iostream>
#include <core/data_model.h>
#include <core/classifier.h>

const std::string file_path_training_images = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/trainingimagesandlabels.txt";
const std::string file_path_test_training_images = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/testimagesandlabels.txt";
const std::string error = "error";

TEST_CASE("Sanity check") {
  REQUIRE(1 < 2);
}

TEST_CASE("Test model accuracy") {
  naivebayes::DataModel model;
  std::ifstream input_file(file_path_training_images);
  if (input_file.is_open()) {
    input_file >> model;
  } else {
    std::cerr << error << std::endl;
  }
  
  model.CalculateModelAccuracy(file_path_test_training_images);
  REQUIRE(model.GetModelAccuracy() > 0.7);
   
  REQUIRE(1 < 2);
}