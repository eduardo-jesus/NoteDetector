#ifndef NOTE_IMG_OBJ
#define NOTE_IMG_OBJ

#include "ImgObject.h"

class NoteImgObject : public ImgObject {
private:
    std::string tag_;
    int value_;
    std::vector<std::vector<cv::Point2f>> patches_;
    std::vector<cv::Point2f> corners_;
public:
    NoteImgObject(void);
    NoteImgObject(std::string tag, std::string filename, int value = 0, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL, std::vector<std::vector<cv::Point2f>> patches = std::vector<std::vector<cv::Point2f>>());
    ~NoteImgObject(void);
    
    void setTag(std::string tag);
    std::string getTag();

    void setValue(int value);
    int getValue();
    std::vector<cv::Point2f>& getCorners();

    void selectKeyPoints(std::vector<cv::KeyPoint> keypoints);
    void detectKeypoints(cv::FeatureDetector* detector);

    static NoteImgObject create5Front(bool with_patches, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL);
    static NoteImgObject create5NFront(bool with_patches, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL);
    static NoteImgObject create5Back(bool with_patches, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL);
    static NoteImgObject create5NBack(bool with_patches, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL);
    static NoteImgObject create10Front(bool with_patches, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL);
    static NoteImgObject create10Back(bool with_patches, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL);
    static NoteImgObject create20Front(bool with_patches, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL);
    static NoteImgObject create20Back(bool with_patches, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL);
    static NoteImgObject create50Front(bool with_patches, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL);
    static NoteImgObject create50Back(bool with_patches, cv::FeatureDetector* detector = NULL, cv::DescriptorExtractor* extractor = NULL);
};

#endif