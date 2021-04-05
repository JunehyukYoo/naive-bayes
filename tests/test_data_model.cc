#include <catch2/catch.hpp>

#include <core/data_model.h>
#include <fstream>
#include <iostream>

using namespace naivebayes;

const std::string file_path = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/testtrainingimages.txt";
const std::string save_file_path = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/savefile.txt";
const std::string error_message = "Could not open file.";

TEST_CASE("Test variables for correct initialization before feeding the model data") {
  DataModel model(3);
  
  SECTION("Number of images per class test") {
    for (size_t i = 0; i < 10; i++) {
      REQUIRE(model.GetNumPerClass(i) == 0);
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
    for (const auto &element : model.GetProbabilities()) {
      for (size_t row = 0; row < 3; row++) {
        for (size_t col = 0; col < 3; col++) {
          REQUIRE(element.second[row][col] == 0.5);
        }
      }
    }
  }
  
  SECTION("Priors test") {
    for (const auto &element : model.GetPriors()) {
      REQUIRE(element.second == 0);
    }
  }
}

TEST_CASE("Test << operator from data, building model") {
  DataModel model(3);
  std::ifstream input_file(file_path);
  if (input_file.is_open()) {
      input_file >> model;
  } else {
      std::cerr << error_message << std::endl;
  }
  
  REQUIRE(model.GetImageDimensions() == 3);
  REQUIRE(model.GetNumTotalImages() == 4);
  REQUIRE(model.GetNumPerClass(0) == 1);
  REQUIRE(model.GetNumPerClass(1) == 2);
  REQUIRE(model.GetNumPerClass(2) == 0);
  REQUIRE(model.GetNumPerClass(3) == 1);
  
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
  
  SECTION("Testing priors") {
   REQUIRE(model.GetPriorFromClass(0) == Approx(0.143).margin(0.001));
   REQUIRE(model.GetPriorFromClass(1) == Approx(0.214).margin(0.001));
   REQUIRE(model.GetPriorFromClass(2) == Approx(0.071).margin(0.001));
   REQUIRE(model.GetPriorFromClass(3) == Approx(0.143).margin(0.001));
  }
}

TEST_CASE("Test << operator from save file") {
  DataModel model1(3);
  std::ifstream input_file(save_file_path);
  if (input_file.is_open()) {
    input_file >> model1;
  } else {
    std::cerr << error_message << std::endl;
  }
  REQUIRE(model1.GetImageDimensions() == 3);
  REQUIRE(model1.GetNumTotalImages() == 4);
  REQUIRE(model1.GetNumPerClass(0) == 1);
  REQUIRE(model1.GetNumPerClass(1) == 2);
  REQUIRE(model1.GetNumPerClass(2) == 0);
  REQUIRE(model1.GetNumPerClass(3) == 1);

  SECTION("Testing class = 0, 4D vector, row -> col -> class -> shade") {
    SECTION("Shaded") {
      REQUIRE(model1.GetRawData()[0][0][0][1] == 1);
      REQUIRE(model1.GetRawData()[0][1][0][1] == 1);
      REQUIRE(model1.GetRawData()[0][2][0][1] == 1);

      REQUIRE(model1.GetRawData()[1][0][0][1] == 1);
      REQUIRE(model1.GetRawData()[1][1][0][1] == 0);
      REQUIRE(model1.GetRawData()[1][2][0][1] == 1);

      REQUIRE(model1.GetRawData()[2][0][0][1] == 1);
      REQUIRE(model1.GetRawData()[2][1][0][1] == 1);
      REQUIRE(model1.GetRawData()[2][2][0][1] == 1);
    }

    SECTION("Shaded") {
      REQUIRE(model1.GetRawData()[0][0][0][0] == 0);
      REQUIRE(model1.GetRawData()[0][1][0][0] == 0);
      REQUIRE(model1.GetRawData()[0][2][0][0] == 0);

      REQUIRE(model1.GetRawData()[1][0][0][0] == 0);
      REQUIRE(model1.GetRawData()[1][1][0][0] == 1);
      REQUIRE(model1.GetRawData()[1][2][0][0] == 0);

      REQUIRE(model1.GetRawData()[2][0][0][0] == 0);
      REQUIRE(model1.GetRawData()[2][1][0][0] == 0);
      REQUIRE(model1.GetRawData()[2][2][0][0] == 0);
    }
  }
    
  SECTION("Testing class = 1, 4D vector, row -> col -> class -> shade") {
    SECTION("Shaded") {
      REQUIRE(model1.GetRawData()[0][0][1][1] == 1);
      REQUIRE(model1.GetRawData()[0][1][1][1] == 2);
      REQUIRE(model1.GetRawData()[0][2][1][1] == 0);

      REQUIRE(model1.GetRawData()[1][0][1][1] == 0);
      REQUIRE(model1.GetRawData()[1][1][1][1] == 2);
      REQUIRE(model1.GetRawData()[1][2][1][1] == 0);

      REQUIRE(model1.GetRawData()[2][0][1][1] == 1);
      REQUIRE(model1.GetRawData()[2][1][1][1] == 2);
      REQUIRE(model1.GetRawData()[2][2][1][1] == 1);
    }

    SECTION("Unshaded") {
      REQUIRE(model1.GetRawData()[0][0][1][0] == 1);
      REQUIRE(model1.GetRawData()[0][1][1][0] == 0);
      REQUIRE(model1.GetRawData()[0][2][1][0] == 2);

      REQUIRE(model1.GetRawData()[1][0][1][0] == 2);
      REQUIRE(model1.GetRawData()[1][1][1][0] == 0);
      REQUIRE(model1.GetRawData()[1][2][1][0] == 2);

      REQUIRE(model1.GetRawData()[2][0][1][0] == 1);
      REQUIRE(model1.GetRawData()[2][1][1][0] == 0);
      REQUIRE(model1.GetRawData()[2][2][1][0] == 1);
    }
  }
  SECTION("Testing class = 3, 4D vector, row -> col -> class -> shade") {
    SECTION("Shaded") {
      REQUIRE(model1.GetRawData()[0][0][3][1] == 1);
      REQUIRE(model1.GetRawData()[0][1][3][1] == 1);
      REQUIRE(model1.GetRawData()[0][2][3][1] == 1);

      REQUIRE(model1.GetRawData()[1][0][3][1] == 0);
      REQUIRE(model1.GetRawData()[1][1][3][1] == 1);
      REQUIRE(model1.GetRawData()[1][2][3][1] == 1);

      REQUIRE(model1.GetRawData()[2][0][3][1] == 1);
      REQUIRE(model1.GetRawData()[2][1][3][1] == 1);
      REQUIRE(model1.GetRawData()[2][2][3][1] == 1);
    }

    SECTION("Unshaded") {
      REQUIRE(model1.GetRawData()[0][0][3][0] == 0);
      REQUIRE(model1.GetRawData()[0][1][3][0] == 0);
      REQUIRE(model1.GetRawData()[0][2][3][0] == 0);

      REQUIRE(model1.GetRawData()[1][0][3][0] == 1);
      REQUIRE(model1.GetRawData()[1][1][3][0] == 0);
      REQUIRE(model1.GetRawData()[1][2][3][0] == 0);

      REQUIRE(model1.GetRawData()[2][0][3][0] == 0);
      REQUIRE(model1.GetRawData()[2][1][3][0] == 0);
      REQUIRE(model1.GetRawData()[2][2][3][0] == 0);
    }
  }
}
