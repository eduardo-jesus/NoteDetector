#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include "ObjectDetector.h"

void getCombination(std::string feature_name, std::string descriptor_name, std::string matcher_name,
                    cv::FeatureDetector* &detector, cv::DescriptorExtractor* &extractor, cv::DescriptorMatcher* &matcher) {
    if (feature_name == "FAST") {
        detector = new cv::FastFeatureDetector();
    } else if (feature_name == "SURF") {
        detector = new cv::SurfFeatureDetector(400);
    } else if(feature_name == "SIFT") {
        detector = new cv::SiftFeatureDetector();
    } else if(feature_name == "ORB") {
        detector = new cv::OrbFeatureDetector();
    }

    if (descriptor_name == "SURF") {
        extractor = new cv::SurfDescriptorExtractor();
    } else if (descriptor_name == "SIFT") {
        extractor = new cv::SiftDescriptorExtractor();
    } else if (descriptor_name == "ORB") {
        extractor = new cv::OrbDescriptorExtractor();
    } else if (descriptor_name == "BRIEF") {
        extractor = new cv::BriefDescriptorExtractor();
    } else if (descriptor_name == "FREAK") {
        extractor = new cv::FREAK();
    }

    if (matcher_name == "FlannBased") {
        matcher = new cv::FlannBasedMatcher();
    } else if (matcher_name == "Bruteforce") {
        matcher = new cv::BFMatcher();
    }
}

int main(int argc, char** argv) {
    bool testing = true;

    cv::FeatureDetector* detector = NULL; 
    cv::DescriptorExtractor* extractor =NULL; 
    cv::DescriptorMatcher* matcher = NULL;

    std::string combinations[11][3] = {
        {"FAST", "SURF",  "FlannBased"},
        {"SURF", "SURF",  "FlannBased"},
        {"FAST", "SIFT",  "FlannBased"},
        {"SIFT", "SIFT",  "FlannBased"},
        {"FAST", "ORB",   "Bruteforce"},
        {"ORB",  "ORB",   "Bruteforce"},
        {"FAST", "BRIEF", "Bruteforce"},
        {"ORB",  "BRIEF", "Bruteforce"},
        {"FAST", "FREAK", "Bruteforce"},
        {"SURF", "FREAK", "Bruteforce"},
        {"SURF", "SURF",  "Bruteforce"}
    };

    std::string filename = (argc == 2) ? argv[1] : "notes/notes.png";

    ObjectDetector object_detector = ObjectDetector(filename, detector, extractor, matcher);
    object_detector.loadLibrary();

    if (testing) {
        std::cout << "Feature Detector" << "\t" << "Descriptor Extractor" << "\t" << "Matcher Type" << "\n"; 
        for (int i = 0; i < 10; ++i) {
            std::cout << combinations[i][0] << "\t" << combinations[i][1] << "\t" << combinations[i][2] << "\n"; 
            getCombination(combinations[i][0], combinations[i][1], combinations[i][2],
                           detector, extractor, matcher);

            object_detector.computeAll(detector, extractor, matcher);

            object_detector.findAllObjects(false);

            delete detector;
            detector = NULL;
            delete extractor;
            extractor = NULL;
            delete matcher;
            matcher = NULL;
        }
    } else {
        detector = new cv::SurfFeatureDetector(400);
        extractor = new cv::SurfDescriptorExtractor();
        matcher = new cv::BFMatcher();
    }

    return 0;
}