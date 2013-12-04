#ifndef IMG_OBJECT_H
#define IMG_OBJECT_H

#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

class ImgObject
{
protected:
    cv::Mat img_;
    std::vector<cv::KeyPoint> original_keypoints_;
    cv::Mat original_descriptors_;

    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    
    std::vector<std::vector<cv::Point2f>> patches_;
public:
    ImgObject(void);
    ImgObject(std::string filename, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL);
    ~ImgObject(void);

    cv::Mat& getImg();
    std::vector<cv::KeyPoint>& getKeypoints();
    cv::Mat& getDescriptors();

    virtual void detectKeypoints(cv::FeatureDetector* detector);
    void computeDescriptors(cv::DescriptorExtractor* extractor);
    void compute(cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor);
    void resetKeypoints();
    static std::vector<cv::Point2f> createPatch(int x0, int y0, int x1, int y1);

    void removeKeypointsInsideCountour(std::vector<cv::Point2f> countour);
};

#endif