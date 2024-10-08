#include <visualizer/naive_bayes_app.h>
#include <core/classifier.h>

namespace naivebayes {

namespace visualizer {

NaiveBayesApp::NaiveBayesApp()
    : sketchpad_(glm::vec2(kMargin, kMargin), kImageDimension,
                 kWindowSize - 2 * kMargin) {
  ci::app::setWindowSize((int) kWindowSize, (int) kWindowSize);
  std::ifstream input_file(kImageFilePath);
  
  if (input_file.is_open()) {
    input_file >> data_model_;
  } else {
    std::cerr << "error message" << std::endl;
  }
}

void NaiveBayesApp::draw() {
  ci::Color8u background_color(255, 246, 148);  // light yellow
  ci::gl::clear(background_color);

  sketchpad_.Draw();

  ci::gl::drawStringCentered(
      "Press Delete to clear the sketchpad. Press Enter to make a prediction.",
      glm::vec2(kWindowSize / 2, kMargin / 2), ci::Color("black"));

  ci::gl::drawStringCentered(
      "Prediction: " + std::to_string(current_prediction_),
      glm::vec2(kWindowSize / 2, kWindowSize - kMargin / 2), ci::Color("blue"));
}

void NaiveBayesApp::mouseDown(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NaiveBayesApp::mouseDrag(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NaiveBayesApp::keyDown(ci::app::KeyEvent event) {
  switch (event.getCode()) {
    case ci::app::KeyEvent::KEY_RETURN:
      // ask your classifier to classify the image that's currently drawn on the
      // sketchpad and update current_prediction_
      if (IsEmpty(sketchpad_)) {
        current_prediction_ = -1;
      } else {
        //current_prediction_ = data_model_.ClassifyImage(sketchpad_.GetSketchPad()); 
        current_prediction_ = classifier_.ClassifyImage(sketchpad_.GetSketchPad(), data_model_);
      }
      break;

    case ci::app::KeyEvent::KEY_DELETE:
      sketchpad_.Clear();
      break;
  }
}

bool NaiveBayesApp::IsEmpty(Sketchpad sketchpad) const {
  for (size_t row = 0; row < sketchpad.GetSketchPad().size(); row++) {
    for (size_t col = 0; col < sketchpad.GetSketchPad().size(); col++) {
      if (sketchpad.GetSketchPad().at(row).at(col)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace visualizer

}  // namespace naivebayes
