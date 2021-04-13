#include <core/classifier.h>
#include <iostream>
#include <sstream>
#include <fstream>

namespace naivebayes {
  
size_t Classifier::ClassifyImage(const visualizer::Sketchpad &sketchpad, const DataModel &data_model) {
  std::vector<std::vector<bool>> image = sketchpad.GetSketchPad();
  std::vector<float> likelihood_scores;
  for (size_t i = 0; i < data_model.GetNumOfClasses(); i++) {
    if (data_model.GetPriorFromClass(i) == 0) {
      likelihood_scores.push_back(0);
    } else {
      likelihood_scores.push_back(log(data_model.GetPriorFromClass(i)));
    }
  }

  for (size_t row = 0; row < image.size(); row++) {
    for (size_t col = 0; col < image.at(0).size(); col++) {
      for (size_t i = 0; i < data_model.GetNumOfClasses(); i++) {
        if (image.at(row).at(col)) {
          likelihood_scores.at(i) += log(data_model.GetShadedProbabilities().at(i).at(row).at(col));
        } else {
          likelihood_scores.at(i) += log(data_model.GetUnshadedProbabilities().at(i).at(row).at(col));
        }
      }
    }
  }

  int classification = -10;
  float greatest_prob = -std::numeric_limits<float>::max();
  for (size_t i = 0; i < likelihood_scores.size(); i++) {
    if (likelihood_scores.at(i) > greatest_prob) {
      greatest_prob = likelihood_scores.at(i);
      classification = i;
    }
  }
  return classification;
}

float Classifier::CalculateAccuracy(const std::string &testing_file, const DataModel &data_model) {
  std::ifstream test_file(testing_file);
  if (test_file.is_open()) {
    std::string line;
    size_t type_class;
    size_t count = 1;
    size_t num_total = 0;
    size_t num_right = 0;
    std::vector<float> likelihood_scores(data_model.GetNumOfClasses());
    while (getline(test_file, line)) {
      ReadFileByLine(count, data_model, line, type_class, num_total, num_right, likelihood_scores);
    }
    auto model_accuracy_ = static_cast<float>(num_right/(num_total));
    std::cout << "The model accuracy is: " + std::to_string(model_accuracy_) << std::endl;
    return model_accuracy_;
  } else {
    throw std::invalid_argument("Invalid testing images and labels file.");
  }
}

void Classifier::ReadFileByLine(size_t &count, const DataModel &data_model, const std::string &line, size_t &type_class,
                                size_t &testing_total, size_t &testing_right,
                                std::vector<float> &likelihood_scores) {
  size_t one_image_line_req = data_model.GetImageDimensions() + 1;

  if (count > one_image_line_req) {
    count = 1;
  }
  if (count == 1) {
    try {
      type_class = stoi(line);
    } catch (...) {
      throw std::invalid_argument("Broken Training File");
    }
    for (size_t i = 0; i < data_model.GetNumOfClasses(); i++) {
      likelihood_scores[i] = log(data_model.GetPriorFromClass(i));
    }
    testing_total++;
  } else {
    for (size_t col = 0; col < data_model.GetImageDimensions(); col++) {
      if (line.at(col) == kShadedOne || line.at(col) == kShadedTwo) {
        for (size_t i = 0; i < data_model.GetNumOfClasses(); i++) {
          likelihood_scores[i] += log(data_model.GetShadedProbabilities().at(i)[count - 2][col]);
        }
      } else {
        for (size_t i = 0; i < data_model.GetNumOfClasses(); i++) {
          likelihood_scores[i] += log(data_model.GetUnshadedProbabilities().at(i)[count - 2][col]);
        }
      }
    }
  }

  if (count == one_image_line_req) {
    float greatest = -std::numeric_limits<float>::max();
    size_t class_;
    for (size_t i = 0; i < data_model.GetNumOfClasses(); i++) {
      if (likelihood_scores[i] > greatest) {
        greatest = likelihood_scores[i];
        class_ = i;
      }
    }
    if (class_ == type_class) {
      testing_right++;
    }
  }
  count++;
}

} // namespace naivebayes