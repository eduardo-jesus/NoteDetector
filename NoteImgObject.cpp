#include "opencv2/imgproc/imgproc.hpp"

#include "NoteImgObject.h"


NoteImgObject::NoteImgObject(void) {}

NoteImgObject::NoteImgObject(std::string filename, int value, cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor, std::vector<std::vector<cv::Point2f>> patches) :
ImgObject(filename) {

    value_ = value;

    patches_ = patches;

    corners_ = std::vector<cv::Point2f>(4);
    corners_[0] = cv::Point(0, 0);
    corners_[1] = cv::Point(img_.cols, 0);
    corners_[2] = cv::Point(img_.cols, img_.rows);
    corners_[3] = cv::Point(0, img_.rows);
    
    if(detector != NULL && extractor != NULL) {
        compute(detector, extractor);
    }
}

NoteImgObject::~NoteImgObject(void) {}

void NoteImgObject::setValue(int value) {
    value_ = value;
}

int NoteImgObject::getValue() {
    return value_;
}

std::vector<cv::Point2f>& NoteImgObject::getCorners() {
    return corners_;
}

void NoteImgObject::selectKeyPoints(std::vector<cv::KeyPoint> keypoints) {
    if(patches_.empty()) {
        original_keypoints_ = keypoints;
    }

    for(unsigned int i = 0; i < keypoints.size(); ++i) {
        for(unsigned int j = 0; j < patches_.size(); ++j) {
            if(cv::pointPolygonTest(patches_[j], keypoints[i].pt, false) >= 0) {
                original_keypoints_.push_back(keypoints[i]);
                keypoints.erase(keypoints.begin() + i);
                --i;
                break;
            }
        }
    }
}

void NoteImgObject::detectKeypoints(cv::FeatureDetector* detector) {
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img_, keypoints);
    selectKeyPoints(keypoints);
}