#include <core/data_model.h>
#include <iostream>
#include <fstream>
#include <sstream>

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
    probabilities_[i] = initial_vec_outer;
  }
  num_total_images_ = 0;
  std::vector<std::vector<std::vector<std::vector<size_t>>>> sized_array(image_dimensions_,std::vector<std::vector<std::vector<size_t>>>(image_dimensions_, std::vector<std::vector<size_t>>(kNumOfClasses, std::vector<size_t>(2))));
  raw_data_ = sized_array;
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
    probabilities_[i] = initial_vec_outer;
  }
  num_total_images_ = 0;
  std::vector<std::vector<std::vector<std::vector<size_t>>>> sized_array(image_dimensions_,std::vector<std::vector<std::vector<size_t>>>(image_dimensions_, std::vector<std::vector<size_t>>(10, std::vector<size_t>(2))));
  raw_data_ = sized_array;
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
  while (std::getline(is, line)) {
    if (line == data_model.kSaveTitle) {
      is_save_file = true;
    }
    
    if (!is_save_file) {
      data_model.ProcessData(count, data_model, line, type_class);
    } else {
      data_model.LoadSave(count, data_model, line);
    }
  }
  data_model.UpdatePriors();
  return is;
}

std::ostream& operator<<(std::ostream& os, DataModel& data_model) {
  os << data_model.kSaveTitle << std::endl;
  os << std::to_string(data_model.image_dimensions_) << std::endl;
  os << std::to_string(data_model.num_total_images_) << std::endl;
  
  for (const auto &element : data_model.GetNumClass()) {
    os << std::to_string(element.second);
  }
  os << std::endl;
  
  for (const auto &element : data_model.GetProbabilities()) {
    for (size_t row = 0; row < data_model.image_dimensions_; row++) {
      for (size_t col = 0; col < data_model.image_dimensions_; col++) {
        os << std::to_string(element.second[row][col]) << " ";
      }
    }
    os<<std::endl;
  }
  
  /**
  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      for (size_t class_ = 0; class_ < 10; class_++) {
        for (size_t shade = 0; shade < 2; shade++) {
          os << std::to_string(data_model.GetRawData()[row][col][class_][shade]) << std::endl;
        }
      }
    }
  }
  */
  
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

float DataModel::GetPriorFromClass(size_t class_) const {
  for (const auto &element : priors_) {
    if (element.first == class_) {
      return element.second;
    }
  }
  return -1;
}

void DataModel::ProcessData(size_t &count, DataModel &data_model, std::string &line, size_t &type_class) {
  size_t one_image_line_req = data_model.image_dimensions_ + 1;
  if (count > one_image_line_req) {
    count = 1;
  }
  if (count == 1) {
    //check which class it is, update relevant variables
    //line.erase(line.find_last_not_of(" \n\r\t") + 1);
    type_class = stoi(line);
    data_model.num_total_images_++;
    data_model.IncrementNumClassMap(type_class);
  } else {
    //method to update the raw_data array, pass the (count-2) as the row and the charAt index is the col of the image
    for (size_t col = 0; col < data_model.image_dimensions_; col++) {
      if (line.at(col) == data_model.kShadedOne || line.at(col) == data_model.kShadedTwo) {
        data_model.raw_data_[count - 2][col][type_class][1]++;
      } else {
        data_model.raw_data_[count - 2][col][type_class][0]++;
      }
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
    for (size_t i = 0; i < data_model.kNumOfClasses; i++) {
      data_model.num_class_[i] = stoi(std::to_string(line.at(i)));
    }
  } else if (count >= 5 && count < (5 + data_model.kNumOfClasses)) {
    std::stringstream line_stream(line);
    std::string temp;
    std::vector<std::vector<float>> prob_array(data_model.image_dimensions_, std::vector<float>(data_model.image_dimensions_));
    while (line_stream >> temp) {
      for (size_t i = 0; i < num_elements_per_line; i++) {
        if (i % data_model.image_dimensions_ == 0 && i != 0) {
          row++;
          col = 0;
        }
        prob_array[row][col] = stof(temp);
        col++;
      }
      break;
    }
    /*
    for (size_t i = 0; i < num_elements_per_line; i++, line_stream >> temp) {
      if (i % data_model.image_dimensions_ == 0) {
        row++;
        col = 0;
      }
      prob_array[row][col] = stof(temp);
      col++;
    }
     */
    data_model.probabilities_[count - 5] = prob_array;
  } else if (count >= (5 + data_model.kNumOfClasses)) {
    //UPDATE 4D vector somehow lol
    std::stringstream line_stream(line);
    std::string temp;
    while (line_stream >> temp) {
      for (size_t i = 0; i < 2 * num_elements_per_line; i++) {
        if (i % (2 * data_model.image_dimensions_) == 0 && i != 0) {
          row++;
          col = 0;
        }

        if (i % 2 == 0) {
          data_model.raw_data_[row][col][count - data_model.kNumOfClasses - 5][0] = stoi(temp);
        } else {
          data_model.raw_data_[row][col][count - data_model.kNumOfClasses - 5][1] = stoi(temp);
        }
      }
      break;
    }
    /*
    for (size_t i = 0; i < 2 * num_elements_per_line; i++, line_stream >> temp) {
      if (i % (2 * data_model.image_dimensions_) == 0) {
        row++;
        col = 0;
      }  
      
      if (i % 2 == 0) {
        data_model.raw_data_[row][col][count - data_model.kNumOfClasses - 5][0] = stoi(temp);
      } else {
        data_model.raw_data_[row][col][count - data_model.kNumOfClasses - 5][1] = stoi(temp);
      }
    }
     */
  }
  count++;
}

std::vector<std::vector<std::vector<std::vector<size_t>>>> DataModel::GetRawData() const {
  return raw_data_;
}

std::unordered_map<size_t, std::vector<std::vector<float>>> DataModel::GetProbabilities() const {
  return probabilities_;
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

}  // namespace naivebayes