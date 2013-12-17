#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "NoteImgObject.h"

struct FoundObject {
    FoundObject(std::vector<cv::Point2f> countour, int value, std::string tag) : countour_(countour), value_(value), tag_(tag) {};
    std::vector<cv::Point2f> countour_;
    int value_;
    std::string tag_;
};

class ObjectDetector {
private:
    std::vector<NoteImgObject> object_library_;
    NoteImgObject* object_;
    ImgObject scene_;

    std::string used_algorithms_;
    cv::FeatureDetector* feature_detector_;
    cv::DescriptorExtractor* descriptor_extractor_;
    cv::DescriptorMatcher* descriptor_matcher_;

    std::vector<FoundObject> objects_found_;
public:
    ObjectDetector(void);
    ObjectDetector(std::string scene_filename, cv::FeatureDetector* feature_detector, 
        cv::DescriptorExtractor* descriptor_extractor,
        cv::DescriptorMatcher* descriptor_matcher);
    ~ObjectDetector(void);

    void loadLibrary(bool with_patches);
    void computeAll(std::string used_algorithms, cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor, cv::DescriptorMatcher* matcher);
    bool iterate(bool wait);
    void findAllObjects(bool wait);
    bool allPointsInsideCountour(std::vector<cv::Point2f> countour, std::vector<cv::Point2f> inliers);
    void drawCountourWithText(cv::Mat& img, std::vector<cv::Point2f>& countour, std::string text);
    void drawFoundObject(cv::Mat& img, FoundObject found_object);
};

#endif
