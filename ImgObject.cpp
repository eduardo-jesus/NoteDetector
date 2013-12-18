#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include "ImgObject.h"


ImgObject::ImgObject(void) {}

ImgObject::ImgObject(std::string filename, cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor) {
    img_ = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if(!img_.data) {
        std::cout << " Error reading " << filename << std::endl;
        exit(-1); 
    }

    if(detector != NULL && extractor != NULL) {
        compute(detector, extractor);
    }
}

ImgObject::~ImgObject(void) {}

cv::Mat& ImgObject::getImg() {
    return img_;
}

std::vector<cv::KeyPoint>& ImgObject::getKeypoints() {
    return keypoints_;
}

cv::Mat& ImgObject::getDescriptors() {
    return descriptors_;
}

// Detects the image keypoints with the given algorithm
void ImgObject::detectKeypoints(cv::FeatureDetector* detector) {
    detector->detect(img_, original_keypoints_);
}

// Extracts the descriptors from keypoints with the given algorithm
void ImgObject::computeDescriptors(cv::DescriptorExtractor* extractor) {
    extractor->compute(img_, original_keypoints_, original_descriptors_);
}

// Detects the keypoints and extracts the descriptors with the given algorithms
void ImgObject::compute(cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor) {
    detectKeypoints(detector);
    computeDescriptors(extractor);
    resetKeypoints();
}

// Reset the keypoints and descriptors to the original values
void ImgObject::resetKeypoints() {
    keypoints_ = original_keypoints_;
    descriptors_ = original_descriptors_.clone();
}

// Return a vector with the corners of a patch
std::vector<cv::Point2f> ImgObject::createPatch(int x0, int y0, int x1, int y1) {
    std::vector<cv::Point2f> patch(4);
    patch[0] = cv::Point(x0, y0);
    patch[1] = cv::Point(x1, y0);
    patch[2] = cv::Point(x1, y1);
    patch[3] = cv::Point(x0, y1);

    return patch;
}

// Removes the image keypoints that are within the given contour
void ImgObject::removeKeypointsInsideCountour(std::vector<cv::Point2f> countour) {
    cv::Mat new_descriptors;
    unsigned int j = 0;
    for(unsigned int i = 0; i < keypoints_.size(); ++i, ++j) {
        // if the point is inside the countour, remove it from the keypoints vector
        if(cv::pointPolygonTest(countour, keypoints_[i].pt, false) >= 0) {
            keypoints_.erase(keypoints_.begin() + i);
            i--;
        }
        // otherwise, put in new_descriptors the descriptor correspondent to that point
        else {
            new_descriptors.push_back(descriptors_.row(j));
        }
    }
    descriptors_ = new_descriptors;
}