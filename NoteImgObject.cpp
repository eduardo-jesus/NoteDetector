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
        return;
    }
    original_keypoints_.clear();
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

NoteImgObject NoteImgObject::create5Front(cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor) {
    std::vector<std::vector<cv::Point2f>> patches;
    patches.push_back(ImgObject::createPatch(7,6,7+17,6+24)); // left top
    patches.push_back(ImgObject::createPatch(8,99,8+22,99+32)); // left bottom
    patches.push_back(ImgObject::createPatch(188,6,188+37,6+60)); // right top
    patches.push_back(ImgObject::createPatch(130,10,130+101,10+118)); // arc
    return NoteImgObject("notes/5eu_r.jpg", 5, detector, extractor, patches);
}

NoteImgObject NoteImgObject::create5Back(cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor) {
    std::vector<std::vector<cv::Point2f>> patches;
    patches.push_back(ImgObject::createPatch(7,7,7+115,7+73)); // left top
    patches.push_back(ImgObject::createPatch(6,100,6+22,100+32)); // left bottom
    patches.push_back(ImgObject::createPatch(239,5,239+18,5+25)); // right top
    patches.push_back(ImgObject::createPatch(241,106,241+15,106+24)); // right bottom
    return NoteImgObject("notes/5eu_v.jpg", 5, detector, extractor, patches);
}

NoteImgObject NoteImgObject::create10Front(cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor) {
    std::vector<std::vector<cv::Point2f>> patches;
    patches.push_back(ImgObject::createPatch(5,5,5+25,5+24)); // left top
    patches.push_back(ImgObject::createPatch(5,104,5+32,104+29)); // left bottom
    patches.push_back(ImgObject::createPatch(169,6,169+59,6+51)); // right top
    patches.push_back(ImgObject::createPatch(135,29,135+96,29+102)); // arc
    return NoteImgObject("notes/10eu_r.jpg", 10, detector, extractor, patches);
}

NoteImgObject NoteImgObject::create10Back(cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor) {
    std::vector<std::vector<cv::Point2f>> patches;
    patches.push_back(ImgObject::createPatch(8,6,8+149,6+47)); // left top
    patches.push_back(ImgObject::createPatch(9,104,9+27,104+30)); // left bottom
    patches.push_back(ImgObject::createPatch(233,4,233+25,4+24)); // right top
    patches.push_back(ImgObject::createPatch(235,110,235+24,110+23)); // right bottom
    return NoteImgObject("notes/10eu_v.jpg", 10, detector, extractor, patches);
}

NoteImgObject NoteImgObject::create20Front(cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor) {
    std::vector<std::vector<cv::Point2f>> patches;
    patches.push_back(ImgObject::createPatch(4,4,4+26,4+23)); // left top
    patches.push_back(ImgObject::createPatch(6,109,6+31,109+27)); // left bottom
    patches.push_back(ImgObject::createPatch(159,5,159+63,5+52)); // right top
    patches.push_back(ImgObject::createPatch(129,45,129+106,45+92)); // arc
    return NoteImgObject("notes/20eu_r.jpg", 20, detector, extractor, patches);
}

NoteImgObject NoteImgObject::create20Back(cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor) {
    std::vector<std::vector<cv::Point2f>> patches;
    patches.push_back(ImgObject::createPatch(6,3,6+150,3+61)); // left top
    patches.push_back(ImgObject::createPatch(7,111,7+32,111+29)); // left bottom
    patches.push_back(ImgObject::createPatch(236,3,236+24,3+26)); // right top
    patches.push_back(ImgObject::createPatch(232,114,232+28,114+25)); // right bottom
    return NoteImgObject("notes/20eu_v.jpg", 20, detector, extractor, patches);
}

NoteImgObject NoteImgObject::create50Front(cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor) {
    std::vector<std::vector<cv::Point2f>> patches;
    patches.push_back(ImgObject::createPatch(3,5,3+24,5+23)); // left top
    patches.push_back(ImgObject::createPatch(3,114,3+32,114+26)); // left bottom
    patches.push_back(ImgObject::createPatch(166,4,166+64,4+52)); // right top
    patches.push_back(ImgObject::createPatch(142,26,142+80,26+105)); // arc
    return NoteImgObject("notes/50eu_r.jpg", 50, detector, extractor, patches);
}

NoteImgObject NoteImgObject::create50Back(cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor) {
    std::vector<std::vector<cv::Point2f>> patches;
    patches.push_back(ImgObject::createPatch(7,5,7+160,5+57)); // left top
    patches.push_back(ImgObject::createPatch(8,114,8+36,114+29)); // left bottom
    patches.push_back(ImgObject::createPatch(237,4,237+24,4+24)); // right top
    patches.push_back(ImgObject::createPatch(220,111,220+36,111+28)); // right bottom
    return NoteImgObject("notes/50eu_v.jpg", 50, detector, extractor, patches);
}