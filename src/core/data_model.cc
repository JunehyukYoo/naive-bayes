#include <core/data_model.h>
#include <iostream>
#include <fstream>
#include <sstream>

namespace naivebayes {
DataModel::DataModel() {
  image_dimensions_ = kDefaultDimensions;
  for (size_t i = 0; i < kNumOfClasses; i++) {
    num_class_[i] = 0;
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
  std::vector<std::vector<std::vector<std::vector<size_t>>>> sized_array(image_dimensions_,std::vector<std::vector<std::vector<size_t>>>(image_dimensions_, std::vector<std::vector<size_t>>(10, std::vector<size_t>(2))));
  raw_data_ = sized_array;
}

DataModel::DataModel(size_t image_dimensions) {
  image_dimensions_ = image_dimensions;
  for (size_t i = 0; i < kNumOfClasses; i++) {
    num_class_[i] = 0;
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
  std::string line;
  size_t count = 1;
  size_t type_class;
  size_t one_image_line_req = data_model.image_dimensions_ + 1; //29 as default
  while (std::getline(is, line)) {
    if (count > one_image_line_req) {
        count = 1;
    }
    if (count == 1) {
      //check which class it is, update relevant variables
      line.erase(line.find_last_not_of(" \n\r\t") + 1);
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
  return is;
}

std::ostream& operator<<(std::ostream& os, DataModel& data_model) {
  //comment for indent
  
  return os;
}

size_t DataModel::GetImageDimensions() const {
  return image_dimensions_;
}

size_t DataModel::GetNumTotalImages() const {
  return num_total_images_;
}

size_t DataModel::GetNumClass(size_t class_) {
  std::unordered_map<size_t, size_t>::iterator itr;
  for (itr = num_class_.begin(); itr != num_class_.end(); itr++) {
    if (itr->first == class_) {
        return itr->second;
    }
  }
  return 0;
}

std::vector<std::vector<std::vector<std::vector<size_t>>>> DataModel::GetRawData() const {
  return raw_data_;
}


}  // namespace naivebayes