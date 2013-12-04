#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include "ObjectDetector.h"

int main(int argc, char** argv) {
    //ImgObject scene = ImgObject((argc == 2) ? argv[1] : "notes/notes.png");

    cv::FeatureDetector* detector = new cv::SurfFeatureDetector(400);
    cv::DescriptorExtractor* extractor = new cv::SurfDescriptorExtractor();
    cv::DescriptorMatcher* matcher = new cv::BFMatcher();

    ObjectDetector object_detector = ObjectDetector((argc == 2) ? argv[1] : "notes/notes.png", detector, extractor, matcher);
    object_detector.loadLibrary();
    object_detector.findAllObjects();

    return 0;
}