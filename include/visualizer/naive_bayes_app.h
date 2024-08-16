#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "sketchpad.h"
#include "core/data_model.h"
#include "core/classifier.h"

namespace naivebayes {

namespace visualizer {

/**
 * Allows a user to draw a digit on a sketchpad and uses Naive Bayes to
 * classify it.
 */
class NaiveBayesApp : public ci::app::App {
 public:
  NaiveBayesApp();

  void draw() override;
  void mouseDown(ci::app::MouseEvent event) override;
  void mouseDrag(ci::app::MouseEvent event) override;
  void keyDown(ci::app::KeyEvent event) override;
  bool IsEmpty(Sketchpad sketchpad) const;
  
  const double kWindowSize = 875;
  const double kMargin = 100;
  const size_t kImageDimension = 28;
  const std::string kImageFilePath = "/Users/s200808/Documents/Cinder/my-projects/naive-bayes/data/trainingimagesandlabels.txt";

 private:
  Sketchpad sketchpad_;
  DataModel data_model_;
  Classifier classifier_;
  int current_prediction_ = -1;
};

}  // namespace visualizer

}  // namespace naivebayes
