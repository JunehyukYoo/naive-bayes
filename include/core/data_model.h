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
   * Processes data from a training images data set or a data set to test model accuracy.
   * @param count A size_t variable used within logic to decode data sets.
   * @param data_model The data model being read/altered.
   * @param line The line that is passed form the data set.
   * @param type_class The type of class the current image in the data set is.
   */
  void ProcessData(size_t& count, DataModel& data_model, const std::string& line, size_t& type_class);
  
  /**
   * Loads in a save file to read.
   * @param count A size_t variable used within logic to decode the save file.
   * @param data_model The data model being altered.
   * @param line The line that is passed form the save file.
   */
  void LoadSave(size_t& count, DataModel& data_model, std::string& line);
  
  /** Updates priors list */
  void UpdatePriors();
  
  /** Updates probabilities map */
  void UpdateProbabilities();
  
  /**
   * Helper method used in LoadSave() in order to load in probability maps from the save file.
   * @param count A size_t variable used within logic to decode the save file.
   * @param data_model The data model being altered.
   * @param line The line that is passed form the save file.
   * @param shaded A boolean to show whether we are reading the shaded or unshaded probabilities map.
   */
  void LoadProbabilities(size_t &count, DataModel &data_model, const std::string &line, const bool shaded);
  
  /**
   * Increments num_class_ unordered map.
   * @param class_ The class of image who's count is being incremented.
   */
  void IncrementNumClassMap(const size_t& class_);

  /**
   * Returns the number of a certain class type within the data set.
   * @param class_ The class type.
   * @return The number of images that fall under the class type.
   */
  size_t GetNumPerClass(const size_t& class_) const;
  
  /**
   * Returns the prior value of a certain class type.
   * @param class_ The class type.
   * @return The prior of the class type.
   */
  float GetPriorFromClass(const size_t& class_) const;
  
  /**
   * Method to test model accuracy (updates the model_accuracy_ variable)
   * @param test_file_path A string with the path to the testing data set.
   */
  void TestModelAccuracy(const std::string& test_file_path);
  
  /**
   * Calculate the accuracy of the model by taking in a testing images file.
   * @param count A size_t variable used within logic to decode data sets.
   * @param data_model The data model being read/altered.
   * @param line The line that is passed form the data set.
   * @param type_class The type of class the current image in the data set is.
   * @param testing_total The total amount of images in the file.
   * @param testing_right The total amount of images the model guessed right.
   * @param likelihood_scores The likelihood scores holding probabilities for being each class for each image
   */
  void CalculateAccuracy(size_t& count, DataModel& data_model, const std::string& line, size_t& type_class, 
                         size_t& testing_total, size_t& testing_right, std::vector<float>& likelihood_scores);
  
  /**
   * Classifies the image drawn in on the sketchpad.
   * @param image The 2D vector corresponding to an image.
   * @return The model's prediction.
   */
  int ClassifyImage(const std::vector<std::vector<bool>>& image);
  
  /** Getters */
  size_t GetImageDimensions() const;
  size_t GetNumTotalImages() const;
  std::unordered_map<size_t, size_t> GetNumClass() const;
  std::vector<std::vector<std::vector<std::vector<size_t>>>> GetRawData() const;
  std::unordered_map<size_t, std::vector<std::vector<float>>> GetShadedProbabilities() const;
  std::unordered_map<size_t, std::vector<std::vector<float>>> GetUnshadedProbabilities() const;
  std::unordered_map<size_t, float> GetPriors() const;
  float GetModelAccuracy() const;

 private:
  size_t image_dimensions_;
  size_t num_total_images_;
  float model_accuracy_;
  
  /** class -> num of images of class */
  std::unordered_map<size_t, size_t> num_class_;

  /** row -> col -> class -> shaded/unshaded */
  std::vector<std::vector<std::vector<std::vector<size_t>>>> raw_data_;
  
  /** class -> prior */
  std::unordered_map<size_t, float> priors_;

  /** class -> probability for each pixel to be shaded*/
  std::unordered_map<size_t, std::vector<std::vector<float>>> shaded_probabilities_;

  /** class -> probability for each pixel to be unshaded*/
  std::unordered_map<size_t, std::vector<std::vector<float>>> unshaded_probabilities_;

  /** MODEL CONSTANTS */
  const size_t kLaplaceK = 1;
  const size_t kNumOfClasses = 10;
  const size_t kDefaultDimensions = 28;
  const size_t kShadingOptions = 2;
  const size_t kNumOfLinesInSaveFileBeforeAnyVec = 5;
  const char kShadedOne = '#';
  const char kShadedTwo = '+';

  /** SAVE FILE CONSTANTS */
  const std::string kSaveTitle = "SAVE_FILE";
  const std::string kBackupSaveFilePath = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/backupsavefile.txt";
};

}  // namespace naivebayes

