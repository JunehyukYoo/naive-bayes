#include <core/data_model.h>
#include <iostream>
#include <fstream>
#include <sstream>

namespace naivebayes {
DataModel::DataModel() {
  image_dimensions_ = 28;
}

DataModel::DataModel(size_t image_dimensions) {
  image_dimensions_ = image_dimensions;
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
  size_t one_image_line_req = data_model.image_dimensions_ + 1; //29
  while (std::getline(is, line)) {
    if (count > one_image_line_req) {
        count = 1;
    }
    if (count == 1) {
      //check which class it is, update num_total_images and map
      //type_class = class
      line.erase(line.find_last_not_of(" \n\r\t") + 1);
      type_class = stoi(line);
      data_model.num_total_images_++;
      data_model.IncrementNumClassMap(type_class);
    } else {
      //method to update the raw_data array, pass the (count-2) as the row and the charAt index is the col of the image
      //add incremement
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

}  // namespace naivebayes