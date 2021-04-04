#include <string>
#include <unordered_map>
#include <vector>

namespace naivebayes {

class DataModel {
 public:
  DataModel();
  DataModel(size_t image_dimensions);
  friend std::istream& operator>>(std::istream& is, DataModel& data_model);
  void ProcessLine();
  void IncrementNumClassMap(size_t class_);

 private:
  size_t image_dimensions_;
  size_t num_total_images_;
  std::unordered_map<size_t, size_t> num_class_;

  /** row -> col -> class -> shaded/unshaded */
  std::vector<std::vector<std::vector<std::vector<int>>>> raw_data_;

  /** class -> probability for each pixel */
  std::unordered_map<size_t, std::vector<std::vector<float>>> probabilities_;

  const size_t kLaplaceK = 1;
  const size_t kNumOfClasses = 10;
  const char kShadedOne = '#';
  const char kShadedTwo = '+';
};

}  // namespace naivebayes

/*
TODO: rename this file. You'll also need to modify CMakeLists.txt.

You can (and should) create more classes and files in include/core (header
 files) and src/core (source files); this project is too big to only have a
 single class.

Make sure to add any files that you create to CMakeLists.txt.

TODO Delete this comment before submitting your code.
*/
