#include <string>
#include <unordered_map>
#include <vector>

namespace naivebayes {

class DataModel {
 public:
  /**
   * Constructor for the data model that sets the image_dimensions to 28 as default.
  */
  DataModel();
  
  /**
   * Constructor for data model which allows for n x n images.
   * @param image_dimensions The side length of the image.
   */
  DataModel(size_t image_dimensions);
  
  /**
   * Operator that returns an istream that takes in file with images. Parses through the images and updates relevant
   * information.
   * @param is The istream.
   * @param data_model The current data model being fed data.
   * @return The istream.
   */
  friend std::istream& operator>>(std::istream& is, DataModel& data_model);
  
  /**
   * Operator used to write output files to save the current model by manipulating the ostream. 
   * @param os The ostream.
   * @param data_model The current data model being saved.
   * @return The ostream.
   */
  friend std::ostream& operator<<(std::ostream& os, DataModel& data_model);
  
  /**
   * Process a line to load data from a text file.
   */
  void ProcessData(size_t& count, DataModel& data_model, std::string& line, size_t& type_class);
  
  /** Load in a save file */
  void LoadSave(size_t& count, DataModel& data_model, std::string& line);
  
  /** Updates priors list */
  void UpdatePriors();
  
  /** Updates probabilities map */
  void UpdateProbabilities();
  
  /**
   * Increments num_class_ unordered map.
   * @param class_ The class of image who's count is being incremented.
   */
  void IncrementNumClassMap(size_t class_);

  /**
   * Returns the number of a certain class type within the data set.
   * @param class_ The class type.
   * @return The number of images that fall under the class type.
   */
  size_t GetNumPerClass(size_t class_) const;
  
  /**
   * Returns the prior value of a certain class type.
   * @param class_ The class type.
   * @return The prior of the class type.
   */
  float GetPriorFromClass(size_t class_) const;
  
  /** Getters */
  size_t GetImageDimensions() const;
  size_t GetNumTotalImages() const;
  std::unordered_map<size_t, size_t> GetNumClass() const;
  std::vector<std::vector<std::vector<std::vector<size_t>>>> GetRawData() const;
  std::unordered_map<size_t, std::vector<std::vector<float>>> GetProbabilities() const;
  std::unordered_map<size_t, float> GetPriors() const;

 private:
  size_t image_dimensions_;
  size_t num_total_images_;
  
  /** class -> num of images of class */
  std::unordered_map<size_t, size_t> num_class_;

  /** row -> col -> class -> shaded/unshaded */
  std::vector<std::vector<std::vector<std::vector<size_t>>>> raw_data_;
  
  /** class -> prior */
  std::unordered_map<size_t, float> priors_;

  /** class -> probability for each pixel */
  std::unordered_map<size_t, std::vector<std::vector<float>>> probabilities_;

  /** MODEL CONSTANTS */
  const size_t kLaplaceK = 1;
  const size_t kNumOfClasses = 10;
  const size_t kDefaultDimensions = 28;
  const char kShadedOne = '#';
  const char kShadedTwo = '+';

  /** SAVE FILE CONSTANTS */
  const std::string kSaveTitle = "SAVE_FILE";
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
