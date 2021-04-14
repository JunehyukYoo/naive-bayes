#include "data_model.h"

namespace naivebayes {
    
class Classifier {
 public:
  /**
   * Classifies an image of a certain image from a sketchpad.
   * @param sketchpad The sketchpad holding the image for the data
   * @param data_model The data model used to make the prediction
   * @return The class of image as a size_t variable
   */
  static size_t ClassifyImage(const std::vector<std::vector<bool>>& board, const DataModel& data_model);
  
  /**
   * Calculate the accuracy of a model with a images and labels file.
   * @param testing_file The path to the testing file
   * @param data_model The data model to test the accuracy of
   * @return The accuracy as a float
   */
  static float CalculateAccuracy(const std::string& testing_file, const DataModel& data_model);
  
  /**
   * Helper method used to read a file by line when calculating accuracy of a model.
   * @param count A size_t variable used within logic to decode the save file.
   * @param data_model The data model being altered
   * @param line The line that is passed form the save file
   * @param type_class The type of class the current image is
   * @param testing_total The total number of images in the file
   * @param testing_right The total number of images guessed right by the model
   * @param likelihood_scores The likelihood scores of the current image for all classes
   */
  static void ReadFileByLine(size_t& count, const DataModel& data_model, const std::string& line, size_t& type_class,
                      size_t& testing_total, size_t& testing_right, std::vector<float>& curr_likelihood_scores, 
                      const std::vector<float>& likelihood_scores_with_priors);
  
 private:
  static const char kShadedOne = '#';
  static const char kShadedTwo = '+';
};
} // namespace naivebayes