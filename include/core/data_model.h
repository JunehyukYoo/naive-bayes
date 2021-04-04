#include <string>
#include <map>

namespace naivebayes {

class DataModel {
 public:
  std::string GetBestClass() const;
  friend std::istream& operator>>(std::istream& is, DataModel& data_model);

 private:
  size_t num_total_images_;
  std::map<int, int> num_class_;
  const size_t kConstantK = 1;
  const size_t kNumOfClasses = 10;
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
