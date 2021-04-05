#include <catch2/catch.hpp>

#include <core/data_model.h>
#include <fstream>
#include <iostream>

using namespace naivebayes;

const std::string file_path = "data/testtrainingimages.txt";
const std::string error_message = "Could not open file.";

TEST_CASE("Check that vectors and size_t counts update correctly with << operator") {
  DataModel model(3);
  std::ifstream input_file(file_path);
  if (input_file.is_open()) {
      input_file >> model;
  } else {
      std::cerr << error_message << std::endl;
  }
  REQUIRE(model.GetImageDimensions() == 3);
  REQUIRE(model.GetNumTotalImages() == 2);
  REQUIRE(model.GetNumClass(0) == 1);
  REQUIRE(model.GetNumClass(1) == 1);
  REQUIRE(model.GetNumClass(2) == 0);
  //Testing class=0
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

