#include <catch2/catch.hpp>

#include <fstream>
#include <iostream>
#include <core/data_model.h>
#include <core/classifier.h>

const std::string file_path_small_training_images = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/smallsettrainingimages.txt";
const std::string file_path_small_test_training_images = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/smallsettesttrainingimages.txt";
const std::string file_path_0 = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/small-images-and-labels-data-sets/testnumber0.txt";
const std::string file_path_1 = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/small-images-and-labels-data-sets/testnumber1.txt";
const std::string file_path_2 = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/small-images-and-labels-data-sets/testnumber2.txt";
const std::string file_path_3 = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/small-images-and-labels-data-sets/testnumber3.txt";
const std::string file_path_4 = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/small-images-and-labels-data-sets/testnumber4.txt";
const std::string file_path_5 = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/small-images-and-labels-data-sets/testnumber5.txt";

const std::string error = "error";

TEST_CASE("Sanity Check: Test model accuracy") {
  naivebayes::Classifier classifier;
  naivebayes::DataModel model(3);
  std::ifstream input_file(file_path_small_training_images);
  if (input_file.is_open()) {
    input_file >> model;
  } else {
    std::cerr << error << std::endl;
  }
  
  classifier.CalculateAccuracy(file_path_small_test_training_images, model);
  REQUIRE(classifier.GetModelAccuracy() > 0.7);
}

TEST_CASE("Verify likelihood scores") {
  naivebayes::DataModel model(3);
  std::ifstream input_file(file_path_small_training_images);
  if (input_file.is_open()) {
    input_file >> model;
  } else {
    std::cerr << error << std::endl;
  }
  SECTION("Number = 0") {
    naivebayes::Classifier classifier;
    classifier.CalculateAccuracy(file_path_0, model);
    std::vector<float> likelihood_scores = classifier.GetCurrentLikelihoodScores();
    REQUIRE(likelihood_scores[0] == Approx(log(0.142857) + 9 * log(0.666667)).margin(0.001));
  }
  SECTION("Number = 1") {
    naivebayes::Classifier classifier;
    classifier.CalculateAccuracy(file_path_1, model);
    std::vector<float> likelihood_scores = classifier.GetCurrentLikelihoodScores();
    REQUIRE(likelihood_scores[1] == Approx(log(0.214286) + 6 * log(0.750000) + 3 * log(0.500000)).margin(0.001));
  }
  SECTION("Number = 2") {
    naivebayes::Classifier classifier;
    classifier.CalculateAccuracy(file_path_2, model);
    std::vector<float> likelihood_scores = classifier.GetCurrentLikelihoodScores();
    REQUIRE(likelihood_scores[2] == Approx(log(0.071429) + 9 * log(0.5)).margin(0.001));
  }
  SECTION("Number = 3") {
    naivebayes::Classifier classifier;
    classifier.CalculateAccuracy(file_path_3, model);
    std::vector<float> likelihood_scores = classifier.GetCurrentLikelihoodScores();
    REQUIRE(likelihood_scores[3] == Approx(log(0.142857) + 9 * log(0.666667)).margin(0.001));
  }
  SECTION("Number = 4") {
    naivebayes::Classifier classifier;
    classifier.CalculateAccuracy(file_path_4, model);
    std::vector<float> likelihood_scores = classifier.GetCurrentLikelihoodScores();
    REQUIRE(likelihood_scores[4] == Approx(log(0.071429) + 9 * log(0.5)).margin(0.001));
  }
  SECTION("Number = 5") {
    naivebayes::Classifier classifier;
    classifier.CalculateAccuracy(file_path_5, model);
    std::vector<float> likelihood_scores = classifier.GetCurrentLikelihoodScores();
    REQUIRE(likelihood_scores[5] == Approx(log(0.071429) + 9 * log(0.5)).margin(0.001));
  }
}