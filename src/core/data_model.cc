#include <core/data_model.h>

namespace naivebayes {

std::string DataModel::GetBestClass() const {
  return "CS 126";
}

std::istream &operator>>(std::istream &is, DataModel &data_model) {
  return is;
}

}  // namespace naivebayes