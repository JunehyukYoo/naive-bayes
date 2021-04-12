#include <core/data_model.h>
#include <iostream>
#include <sstream>
#include <fstream>

namespace naivebayes {
DataModel::DataModel() {
  image_dimensions_ = kDefaultDimensions;
  for (size_t i = 0; i < kNumOfClasses; i++) {
    num_class_[i] = 0;
    priors_[i] = 0;
    std::vector<float> initial_vec_inner;
    std::vector<std::vector<float>> initial_vec_outer;
    for (size_t j = 0; j < image_dimensions_; j++) {
      initial_vec_inner.push_back(0.5);
    }
    for (size_t j = 0; j < image_dimensions_; j++) {
        initial_vec_outer.push_back(initial_vec_inner);
    }
    shaded_probabilities_[i] = initial_vec_outer;
    unshaded_probabilities_[i] = initial_vec_outer;
  }
  num_total_images_ = 0;
  std::vector<std::vector<std::vector<std::vector<size_t>>>> sized_array(image_dimensions_,std::vector<std::vector<std::vector<size_t>>>(image_dimensions_, std::vector<std::vector<size_t>>(kNumOfClasses, std::vector<size_t>(kShadingOptions))));
  raw_data_ = sized_array;
  sized_array.clear();
}

DataModel::DataModel(size_t image_dimensions) {
  image_dimensions_ = image_dimensions;
  for (size_t i = 0; i < kNumOfClasses; i++) {
    num_class_[i] = 0;
    priors_[i] = 0;
    std::vector<float> initial_vec_inner;
    std::vector<std::vector<float>> initial_vec_outer;
    for (size_t j = 0; j < image_dimensions_; j++) {
        initial_vec_inner.push_back(0.5);
    }
    for (size_t j = 0; j < image_dimensions_; j++) {
        initial_vec_outer.push_back(initial_vec_inner);
    }
    shaded_probabilities_[i] = initial_vec_outer;
    unshaded_probabilities_[i] = initial_vec_outer;
  }
  num_total_images_ = 0;
  std::vector<std::vector<std::vector<std::vector<size_t>>>> sized_array(image_dimensions_,std::vector<std::vector<std::vector<size_t>>>(image_dimensions_, std::vector<std::vector<size_t>>(kNumOfClasses, std::vector<size_t>(kShadingOptions))));
  raw_data_ = sized_array;
  sized_array.clear();
}

void DataModel::IncrementNumClassMap(size_t class_) {
  std::unordered_map<size_t, size_t>::iterator itr;
  for (itr = num_class_.begin(); itr != num_class_.end(); itr++) {
    if (itr->first == class_) {
        itr->second++;
    }
  }
}

std::istream &operator>>(std::istream &is, DataModel &data_model) {
  bool is_save_file = false;
  std::string line;
  size_t count = 1;
  size_t type_class;
  size_t unused_total = 0;
  size_t unused_right = 0;
  std::vector<float> unused_vec;

  std::ofstream output_file(data_model.kBackupSaveFilePath);
  if (output_file.is_open()) {
    output_file << data_model;
  } else {
    std::cerr << "Backup file cannot be opened." << std::endl;
  }
  
  while (std::getline(is, line)) {
    if (line == data_model.kSaveTitle && count == 1) {
      is_save_file = true;
    }
    
    if (!is_save_file) {
      data_model.ProcessData(count, data_model, line, type_class, false, unused_total, unused_right, unused_vec);
      data_model.UpdatePriors();
      data_model.UpdateProbabilities();
    } else {
      data_model.LoadSave(count, data_model, line);
    }
  }
  
  if (is_save_file && count < 5 + 3 * data_model.kNumOfClasses) {
      std::ifstream backup_file(data_model.kBackupSaveFilePath);
      if (backup_file.is_open()) {
          backup_file >> data_model;
      } else {
          std::cerr << "Backup file cannot be opened." << std::endl;
      }
    throw std::invalid_argument("Bad save file");
  }
  return is;
}

std::ostream& operator<<(std::ostream& os, DataModel& data_model) {
  os << data_model.kSaveTitle << std::endl;
  os << std::to_string(data_model.image_dimensions_) << std::endl;
  os << std::to_string(data_model.num_total_images_) << std::endl;
  
  for (const auto &element : data_model.GetNumClass()) {
    os << std::to_string(element.second) << " ";
  }
  os << std::endl;
  
  for (const auto &element : data_model.GetShadedProbabilities()) {
    for (size_t row = 0; row < data_model.image_dimensions_; row++) {
      for (size_t col = 0; col < data_model.image_dimensions_; col++) {
        os << std::to_string(element.second[row][col]) << " ";
      }
    }
    os<<std::endl;
  }

  for (const auto &element : data_model.GetUnshadedProbabilities()) {
    for (size_t row = 0; row < data_model.image_dimensions_; row++) {
      for (size_t col = 0; col < data_model.image_dimensions_; col++) {
        os << std::to_string(element.second[row][col]) << " ";
      }
    }
    os<<std::endl;
  }
  
  for (size_t class_ = 0; class_ < 10; class_++) {
    for (size_t row = 0; row < 3; row++) {
      for (size_t col = 0; col < 3; col++) {
        for (size_t shade = 0; shade < 2; shade++) {
          os << std::to_string(data_model.GetRawData()[row][col][class_][shade]) << " ";
        }
      }
    }
    os << std::endl;
  }
  
  for (const auto &element : data_model.GetPriors()) {
    os << std::to_string(element.second) << " ";
  }
  return os;
}

size_t DataModel::GetNumPerClass(size_t class_) const {
  std::unordered_map<size_t, size_t>::const_iterator itr;
  for (itr = num_class_.begin(); itr != num_class_.end(); itr++) {
    if (itr->first == class_) {
        return itr->second;
    }
  }
  return 0;
}

void DataModel::UpdatePriors() {
  for (auto &element : priors_) {
    auto numerator = static_cast<float>(kLaplaceK + GetNumPerClass(element.first));
    auto denominator = static_cast<float>(kLaplaceK * kNumOfClasses + num_total_images_);
    element.second = numerator/denominator;
  }
}

void DataModel::UpdateProbabilities() {
  for (auto &element : shaded_probabilities_) {
    for (size_t row = 0; row < image_dimensions_; row++) {
      for (size_t col = 0; col < image_dimensions_; col++) {
        auto numerator_shaded = static_cast<float>(kLaplaceK + raw_data_[row][col][element.first][1]);
        auto denominator = static_cast<float>(2 * kLaplaceK + GetNumPerClass(element.first));
        element.second[row][col] = numerator_shaded / denominator;
      }
    }
  }
  for (auto &element : unshaded_probabilities_) {
    for (size_t row = 0; row < image_dimensions_; row++) {
      for (size_t col = 0; col < image_dimensions_; col++) {
        auto numerator_unshaded = static_cast<float>(kLaplaceK + raw_data_[row][col][element.first][0]);
        auto denominator = static_cast<float>(2 * kLaplaceK + GetNumPerClass(element.first));
        element.second[row][col] = numerator_unshaded / denominator;
      }
    }
  }
}

float DataModel::GetPriorFromClass(size_t class_) const {
  for (const auto &element : priors_) {
    if (element.first == class_) {
      return element.second;
    }
  }
  return -1;
}

void DataModel::ProcessData(size_t &count, DataModel &data_model, std::string &line, size_t &type_class, bool is_test, size_t &testing_total, size_t &testing_right, std::vector<float> &likelihood_scores) {
  size_t one_image_line_req = data_model.image_dimensions_ + 1;
  
  if (count > one_image_line_req) {
    count = 1;
  }
  if (count == 1) {
    //check which class it is, update relevant variables
    try {
      type_class = stoi(line);
    } catch (...) {
      throw std::invalid_argument("Broken Training File");
    }
    if (is_test) {
      for (size_t i = 0; i < data_model.kNumOfClasses; i++) {
        likelihood_scores[i] = log(data_model.GetPriorFromClass(i));
      }
      testing_total++;
    } else {
      data_model.num_total_images_++;
      data_model.IncrementNumClassMap(type_class);
    }
  } else {
    //method to update the raw_data array, pass the (count-2) as the row and the charAt index is the col of the image
    for (size_t col = 0; col < data_model.image_dimensions_; col++) {
      if (line.at(col) == data_model.kShadedOne || line.at(col) == data_model.kShadedTwo) {
        if (is_test) {
          for (size_t i = 0; i < data_model.kNumOfClasses; i++) {
            likelihood_scores[i] += log(data_model.GetShadedProbabilities().at(i)[count - 2][col]);
          }
        } else {
          data_model.raw_data_[count - 2][col][type_class][1]++; 
        }
      } else {
        if (is_test) {
          for (size_t i = 0; i < data_model.kNumOfClasses; i++) {
            likelihood_scores[i] += log(data_model.GetUnshadedProbabilities().at(i)[count - 2][col]);
          }
        } else {
          data_model.raw_data_[count - 2][col][type_class][0]++; 
        }
      }
    }
  }
  
  if (is_test && count == one_image_line_req) {
    float greatest = -std::numeric_limits<float>::max();
    size_t class_;
    for (size_t i = 0; i < data_model.kNumOfClasses; i++) {
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

void DataModel::LoadSave(size_t &count, DataModel &data_model, std::string &line) {
  size_t num_elements_per_line = data_model.image_dimensions_ * data_model.image_dimensions_;
  size_t row = 0;
  size_t col = 0;
  if (count == 2) {
    data_model.image_dimensions_ = stoi(line);
  } else if (count == 3) {
    data_model.num_total_images_ = stoi(line);
  } else if (count == 4) {
    //update count of images per class
    std::stringstream line_stream(line);
    std::string temp;
    size_t class_ = 0;
    while (line_stream >> temp && class_ <= data_model.kNumOfClasses) {
      try {
        data_model.num_class_[class_] = stoi(temp);
      } catch (...) {
        throw std::invalid_argument("Broken Save File");
      }
      class_++;
    }
  } else if (count >= kNumOfLinesInSaveFileBeforeAnyVec && count < (kNumOfLinesInSaveFileBeforeAnyVec + data_model.kNumOfClasses)) {
    //update shaded probabilities map
    data_model.LoadProbabilities(count, data_model, line, true);
  } else if (count >= (kNumOfLinesInSaveFileBeforeAnyVec + data_model.kNumOfClasses) && count < (kNumOfLinesInSaveFileBeforeAnyVec + 2 * data_model.kNumOfClasses)) {
    //update unshaded probabilities map
    data_model.LoadProbabilities(count, data_model, line, false);
  } else if (count >= (kNumOfLinesInSaveFileBeforeAnyVec + 2 * data_model.kNumOfClasses) && count < (kNumOfLinesInSaveFileBeforeAnyVec + 3 * data_model.kNumOfClasses)) {
    //update raw data 4D vector
    std::stringstream line_stream(line);
    std::string temp;
    size_t i = 0;
    while (line_stream >> temp && i < (2 * num_elements_per_line)) {
      if (i % (2 * data_model.image_dimensions_) == 0 && i != 0) {
        row++;
        col = 0;
      }

      if (i % 2 == 0) {
        try {
          data_model.raw_data_[row][col][count - 2 * data_model.kNumOfClasses - 5][0] = stoi(temp);
        } catch (...) {
          throw std::invalid_argument("Broken Save File");
        }
      } else {
        try {
          data_model.raw_data_[row][col][count - 2 * data_model.kNumOfClasses - 5][1] = stoi(temp);
        } catch (...) {
          throw std::invalid_argument("Broken Save File");
        }
      }
      
      if (i % 2 != 0) {
        col++;
      }
      i++;
    }
  } else if (count >= (5 + 3 * data_model.kNumOfClasses)) {
    //update priors map
    std::stringstream line_stream(line);
    std::string temp;
    size_t i = 0;
    while (line_stream >> temp && i <= data_model.kNumOfClasses) {
      try {
        data_model.priors_[i] = stof(temp);
      } catch (...) {
        throw std::invalid_argument("Broken Save File");
      }
      i++;
    }
  }
  count++;
}

std::vector<std::vector<std::vector<std::vector<size_t>>>> DataModel::GetRawData() const {
  return raw_data_;
}

std::unordered_map<size_t, std::vector<std::vector<float>>> DataModel::GetShadedProbabilities() const {
  return shaded_probabilities_;
}

std::unordered_map<size_t, size_t> DataModel::GetNumClass() const {
  return num_class_;
}

std::unordered_map<size_t, float> DataModel::GetPriors() const {
  return priors_;
}
size_t DataModel::GetImageDimensions() const {
  return image_dimensions_;
}

size_t DataModel::GetNumTotalImages() const {
  return num_total_images_;
}

std::unordered_map<size_t, std::vector<std::vector<float>>> DataModel::GetUnshadedProbabilities() const {
  return unshaded_probabilities_;
}

void DataModel::LoadProbabilities(size_t &count, DataModel &data_model, std::string &line, bool shaded) {
  size_t row = 0;
  size_t col = 0;
  std::stringstream line_stream(line);
  std::string temp;
  std::vector<std::vector<float>> prob_array(data_model.image_dimensions_, std::vector<float>(data_model.image_dimensions_));
  size_t i = 0;
  while (line_stream >> temp && i <= data_model.kDefaultDimensions) {
    if (i % data_model.image_dimensions_ == 0 && i != 0) {
      row++;
      col = 0;
    }
    try {
      prob_array[row][col] = stof(temp);
    } catch (...) {
      throw std::invalid_argument("Broken Save File");
    }
    col++;
    i++;
  }
  if (shaded) {
    data_model.shaded_probabilities_[count - 5] = prob_array;
  } else {
    data_model.unshaded_probabilities_[count - (5 + data_model.kNumOfClasses)] = prob_array; 
  }
}

void DataModel::TestModelAccuracy(std::string test_file_path) {
  std::ifstream test_file(test_file_path);
  if (test_file.is_open()) {
    std::string line;
    size_t type_class;
    size_t count = 1;
    size_t num_total = 0;
    size_t num_right = 0;
    std::vector<float> likelihood_scores(kNumOfClasses);
    DataModel temp = *this;
    while (getline(test_file, line)) {
      ProcessData(count, temp, line, type_class, true, num_total, num_right, likelihood_scores); 
    }
    this->model_accuracy_ = static_cast<float>(num_right/(num_total));
    std::cout << "The model accuracy is: " + std::to_string(model_accuracy_) << std::endl;
  } else {
    throw std::invalid_argument("Invalid testing images and labels file.");
  }
}

float DataModel::GetModelAccuracy() const {
  return model_accuracy_;
}

int DataModel::ClassifyImage(std::vector<std::vector<bool>> image) {
  std::vector<float> likelihood_scores;
  for (size_t i = 0; i < kNumOfClasses; i++) {
    if (GetPriorFromClass(i) == 0) {
      likelihood_scores.push_back(0);
    } else {
      likelihood_scores.push_back(log(GetPriorFromClass(i)));
    }
  }
  
  for (size_t row = 0; row < image.size(); row++) {
    for (size_t col = 0; col < image.at(0).size(); col++) {
      for (size_t i = 0; i < kNumOfClasses; i++) {
        if (image.at(row).at(col)) {
          likelihood_scores.at(i) += log(shaded_probabilities_.at(i).at(row).at(col));
        } else {
          likelihood_scores.at(i) += log(unshaded_probabilities_.at(i).at(row).at(col));
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

}  // namespace naivebayes