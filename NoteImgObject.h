#ifndef NOTE_IMG_OBJ
#define NOTE_IMG_OBJ

#include "ImgObject.h"

class NoteImgObject : public ImgObject {
private:
    int value_;
    std::vector<std::vector<cv::Point2f>> patches_;
    std::vector<cv::Point2f> corners_;
public:
    NoteImgObject(void);
    NoteImgObject(std::string filename, int value =0, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL, std::vector<std::vector<cv::Point2f>> patches = std::vector<std::vector<cv::Point2f>>());
    ~NoteImgObject(void);
    
    void setValue(int value);
    int getValue();
    std::vector<cv::Point2f>& getCorners();

    void selectKeyPoints(std::vector<cv::KeyPoint> keypoints);
    void detectKeypoints(cv::FeatureDetector* detector);
};

#endif