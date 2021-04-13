#include "visualizer/sketchpad.h"
#include "data_model.h"

namespace naivebayes {
    
class Classifier {
 public:
  static size_t ClassifyImage(const visualizer::Sketchpad& sketchpad, const DataModel& data_model);
  
  static float CalculateAccuracy(const std::string& testing_file, const DataModel& data_model);
  
  static void ReadFileByLine(size_t &count, const DataModel &data_model, const std::string &line, size_t &type_class,
                      size_t &testing_total, size_t &testing_right,
                      std::vector<float> &likelihood_scores);
 private:
  static const char kShadedOne = '#';
  static const char kShadedTwo = '+';
};
} // namespace naivebayes