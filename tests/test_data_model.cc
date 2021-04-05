#include <catch2/catch.hpp>

#include <core/data_model.h>
#include <fstream>
#include <iostream>

using namespace naivebayes;

const std::string file_path = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/testtrainingimages.txt";
const std::string error_message = "Could not open file.";

TEST_CASE("Test variables for correct initialization before feeding the model data") {
  DataModel model(3);
  
  SECTION("Number of images per class test") {
    for (size_t i = 0; i < 10; i++) {
      REQUIRE(model.GetNumClass(i) == 0);
    }
  }
  
  SECTION("4D vector test") {
    for (size_t row = 0; row < 3; row++) {
      for (size_t col = 0; col < 3; col++) {
        for (size_t class_ = 0; class_ < 10; class_++) {
          for (size_t shade = 0; shade < 2; shade++) {
            REQUIRE(model.GetRawData()[row][col][class_][shade] == 0);
          }
        }
      }
    }
  }
  
  SECTION("Probabilities test") {
    std::unordered_map<size_t, std::vector<std::vector<float>>>::const_iterator itr;
    for (itr = model.GetProbabilities().begin(); itr != model.GetProbabilities().end(); itr++) {
      for (size_t row = 0; row < 3; row++) {
        for (size_t col = 0; col < 3; col++) {
          std::vector<std::vector<float>> prob = itr->second;
          REQUIRE(prob[row][col] == 0.5);
        }
      }
    }
  }
}

TEST_CASE("Test << operator, raw data (4 dimensional vector)") {
  DataModel model(3);
  std::ifstream input_file(file_path);
  if (input_file.is_open()) {
      input_file >> model;
  } else {
      std::cerr << error_message << std::endl;
  }
  
  REQUIRE(model.GetImageDimensions() == 3);
  //REQUIRE(model.GetNumTotalImages() == 4);
  REQUIRE(model.GetNumClass(0) == 1);
  REQUIRE(model.GetNumClass(1) == 2);
  REQUIRE(model.GetNumClass(2) == 0);
  REQUIRE(model.GetNumClass(3) == 1);
  
  SECTION("Testing class = 0, 4D vector, row -> col -> class -> shade") {
    SECTION("Shaded") {
      REQUIRE(model.GetRawData()[0][0][0][1] == 1);
      REQUIRE(model.GetRawData()[0][1][0][1] == 1);
      REQUIRE(model.GetRawData()[0][2][0][1] == 1);

      REQUIRE(model.GetRawData()[1][0][0][1] == 1);
      REQUIRE(model.GetRawData()[1][1][0][1] == 0);
      REQUIRE(model.GetRawData()[1][2][0][1] == 1);

      REQUIRE(model.GetRawData()[2][0][0][1] == 1);
      REQUIRE(model.GetRawData()[2][1][0][1] == 1);
      REQUIRE(model.GetRawData()[2][2][0][1] == 1);
    }

    SECTION("Shaded") {
      REQUIRE(model.GetRawData()[0][0][0][0] == 0);
      REQUIRE(model.GetRawData()[0][1][0][0] == 0);
      REQUIRE(model.GetRawData()[0][2][0][0] == 0);

      REQUIRE(model.GetRawData()[1][0][0][0] == 0);
      REQUIRE(model.GetRawData()[1][1][0][0] == 1);
      REQUIRE(model.GetRawData()[1][2][0][0] == 0);

      REQUIRE(model.GetRawData()[2][0][0][0] == 0);
      REQUIRE(model.GetRawData()[2][1][0][0] == 0);
      REQUIRE(model.GetRawData()[2][2][0][0] == 0);
    }
  }
  
  SECTION("Testing class = 1, 4D vector, row -> col -> class -> shade") {
    SECTION("Shaded") {
      REQUIRE(model.GetRawData()[0][0][1][1] == 1);
      REQUIRE(model.GetRawData()[0][1][1][1] == 2);
      REQUIRE(model.GetRawData()[0][2][1][1] == 0);

      REQUIRE(model.GetRawData()[1][0][1][1] == 0);
      REQUIRE(model.GetRawData()[1][1][1][1] == 2);
      REQUIRE(model.GetRawData()[1][2][1][1] == 0);

      REQUIRE(model.GetRawData()[2][0][1][1] == 1);
      REQUIRE(model.GetRawData()[2][1][1][1] == 2);
      REQUIRE(model.GetRawData()[2][2][1][1] == 1);
    }
    
    SECTION("Unshaded") {
      REQUIRE(model.GetRawData()[0][0][1][0] == 1);
      REQUIRE(model.GetRawData()[0][1][1][0] == 0);
      REQUIRE(model.GetRawData()[0][2][1][0] == 2);

      REQUIRE(model.GetRawData()[1][0][1][0] == 2);
      REQUIRE(model.GetRawData()[1][1][1][0] == 0);
      REQUIRE(model.GetRawData()[1][2][1][0] == 2);

      REQUIRE(model.GetRawData()[2][0][1][0] == 1);
      REQUIRE(model.GetRawData()[2][1][1][0] == 0);
      REQUIRE(model.GetRawData()[2][2][1][0] == 1);
    }
  }
  SECTION("Testing class = 3, 4D vector, row -> col -> class -> shade") {
    SECTION("Shaded") {
      REQUIRE(model.GetRawData()[0][0][3][1] == 1);
      REQUIRE(model.GetRawData()[0][1][3][1] == 1);
      REQUIRE(model.GetRawData()[0][2][3][1] == 1);

      REQUIRE(model.GetRawData()[1][0][3][1] == 0);
      REQUIRE(model.GetRawData()[1][1][3][1] == 1);
      REQUIRE(model.GetRawData()[1][2][3][1] == 1);

      REQUIRE(model.GetRawData()[2][0][3][1] == 1);
      REQUIRE(model.GetRawData()[2][1][3][1] == 1);
      REQUIRE(model.GetRawData()[2][2][3][1] == 1);
    }

    SECTION("Unshaded") {
      REQUIRE(model.GetRawData()[0][0][3][0] == 0);
      REQUIRE(model.GetRawData()[0][1][3][0] == 0);
      REQUIRE(model.GetRawData()[0][2][3][0] == 0);

      REQUIRE(model.GetRawData()[1][0][3][0] == 1);
      REQUIRE(model.GetRawData()[1][1][3][0] == 0);
      REQUIRE(model.GetRawData()[1][2][3][0] == 0);

      REQUIRE(model.GetRawData()[2][0][3][0] == 0);
      REQUIRE(model.GetRawData()[2][1][3][0] == 0);
      REQUIRE(model.GetRawData()[2][2][3][0] == 0);
    }
  }
}
