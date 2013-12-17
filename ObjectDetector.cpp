#include <iostream>
#include <sstream>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "ObjectDetector.h"

#include "Log.h"

#define FONT_FACE cv::FONT_HERSHEY_SCRIPT_COMPLEX
#define FONT_THICKNESS 3
#define FONT_RATIO 4

ObjectDetector::ObjectDetector() {}

ObjectDetector::ObjectDetector(std::string scene_filename, cv::FeatureDetector* feature_detector, 
        cv::DescriptorExtractor* descriptor_extractor,
        cv::DescriptorMatcher* descriptor_matcher) {
         
    feature_detector_ = feature_detector;
    descriptor_extractor_ = descriptor_extractor;
    descriptor_matcher_ = descriptor_matcher;

    scene_ = ImgObject(scene_filename, feature_detector_, descriptor_extractor_);
}

ObjectDetector::~ObjectDetector(void) {}

void ObjectDetector::loadLibrary(bool with_patches) {
    object_library_.push_back(NoteImgObject::create5Front(with_patches, feature_detector_, descriptor_extractor_));
    object_library_.push_back(NoteImgObject::create5Back(with_patches, feature_detector_, descriptor_extractor_));
    object_library_.push_back(NoteImgObject::create5NFront(with_patches, feature_detector_, descriptor_extractor_));
    object_library_.push_back(NoteImgObject::create5NBack(with_patches, feature_detector_, descriptor_extractor_));
    object_library_.push_back(NoteImgObject::create10Front(with_patches, feature_detector_, descriptor_extractor_));
    object_library_.push_back(NoteImgObject::create10Back(with_patches, feature_detector_, descriptor_extractor_));
    object_library_.push_back(NoteImgObject::create20Front(with_patches, feature_detector_, descriptor_extractor_));
    object_library_.push_back(NoteImgObject::create20Back(with_patches, feature_detector_, descriptor_extractor_));
    object_library_.push_back(NoteImgObject::create50Front(with_patches, feature_detector_, descriptor_extractor_));
    object_library_.push_back(NoteImgObject::create50Back(with_patches, feature_detector_, descriptor_extractor_));
}

bool ObjectDetector::iterate(bool wait) {
    if (scene_.getDescriptors().rows == 0) {
        Log::instance().debug("\tNo descriptors left.\n_________________________________________\n");
        return false;
    }

    std::vector<cv::DMatch> matches, good_matches;

    //std::cout << scene_.getDescriptors().rows << "\n";
    
    descriptor_matcher_->match(object_->getDescriptors(), scene_.getDescriptors(), matches);
    std::stringstream ss;
    ss << "\tMatches: " << matches.size() << "\n";
    Log::instance().debug(ss.str());
    ss.str("");
    double max_dist = 0;
    double min_dist = 100;
    for(unsigned int i = 0; i < /*object_->getKeypoints().size()*/matches.size(); ++i) {
        double dist = matches[i].distance;
        if(dist < min_dist) {
            min_dist = dist;
        }
        if(dist > max_dist) {
            max_dist = dist;
        }
    }

    for(unsigned int i = 0; i < /*object_->getKeypoints().size()*/matches.size(); ++i) {
        if(matches[i].distance < 3 * min_dist) {
            good_matches.push_back(matches[i]);
        }
    }

    ss << "\tGood matches: " << good_matches.size() << "\n";
    Log::instance().debug(ss.str());
    ss.str("");

    if(good_matches.size() < 4) {
        ss << "\tNeeded 4 points to calculate homography. Have " << good_matches.size() << "\n______________________________________\n";
        Log::instance().debug(ss.str());
        ss.str("");
        cv::Mat img_matches;
        drawMatches( object_->getImg(), object_->getKeypoints(), scene_.getImg(), scene_.getKeypoints(),
            good_matches, img_matches,cv::Scalar::all(-1), cv::Scalar(0,0,255));
        if (wait) {
            cv::imshow(used_algorithms_ + " - Iteration", img_matches);
            cv::waitKey(0);
        }
        

        return false;
    }

    std::vector<cv::Point2f> points_obj, points_scene;
    for(unsigned int i = 0; i < good_matches.size(); ++i) {
        points_obj.push_back(object_->getKeypoints()[good_matches[i].queryIdx].pt);
        points_scene.push_back(scene_.getKeypoints()[good_matches[i].trainIdx].pt);
    }

    cv::Mat inliers;
    cv::Mat homography = cv::findHomography(points_obj, points_scene, cv::RANSAC, 3, inliers);

    std::vector<cv::DMatch> inlier_matches;
    std::vector<cv::Point2f> inlier_points;
    for(int i = 0; i < inliers.rows; ++i) {
        if(inliers.at<uchar>(i, 0) != 0) {
            inlier_matches.push_back(good_matches[i]);
            cv::Point2f point = scene_.getKeypoints()[good_matches[i].trainIdx].pt; 
            inlier_points.push_back(point);
        }
    }

    ss << "\tInlier points: " << inlier_points.size() << "\n";
    Log::instance().debug(ss.str());
    ss.str("");

    cv::Mat img_matches;
    drawMatches(object_->getImg(), object_->getKeypoints(), scene_.getImg(), scene_.getKeypoints(),
        inlier_matches, img_matches,cv::Scalar::all(-1), cv::Scalar(0,0,255));

    std::vector<cv::Point2f> scene_corners(4);
    cv::perspectiveTransform(object_->getCorners(), scene_corners, homography);

    cv::Point2f offset((float) object_->getImg().cols, 0);
    line(img_matches, scene_corners[0] + offset, scene_corners[1] + offset, cv::Scalar(0, 255, 0), 4 );
    line(img_matches, scene_corners[1] + offset, scene_corners[2] + offset, cv::Scalar(0, 255, 0), 4 );
    line(img_matches, scene_corners[2] + offset, scene_corners[3] + offset, cv::Scalar(0, 255, 0), 4 );
    line(img_matches, scene_corners[3] + offset, scene_corners[0] + offset, cv::Scalar(0, 255, 0), 4 );

    if (wait) {
        imshow(used_algorithms_ + " - Iteration", img_matches);
        cv::waitKey(0);
    }

    if(!allPointsInsideCountour(scene_corners, inlier_points)) {
        Log::instance().debug("\tInlier outside contour\n__________________________________________________\n");
        return false;
    }

    scene_.removeKeypointsInsideCountour(scene_corners);

    objects_found_.push_back(FoundObject(scene_corners, object_->getValue(), object_->getTag()));

    Log::instance().debug("\tFound: " + object_->getTag() + "\n_____________________________________\n");

    return true;
}

void ObjectDetector::findAllObjects(bool wait) {
    for(unsigned int i = 0; i < object_library_.size(); ++i) {
        object_ = &object_library_[i];
        Log::instance().debug(object_->getTag() + "\n");
        while(iterate(wait));
        scene_.resetKeypoints();
    }

    cv::Mat img_to_show;
    int total = 0;
    cv::cvtColor(scene_.getImg(), img_to_show, CV_GRAY2RGB);
    for(unsigned int i = 0; i < objects_found_.size(); ++i) {
        drawFoundObject(img_to_show, objects_found_[i]);
        total += objects_found_[i].value_;
    }

    std::stringstream ss;
    ss << total;

    Log::instance().debug("Total amount: " + ss.str() + "\n");

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(ss.str(), FONT_FACE, 1, FONT_THICKNESS, &baseline);
    cv::putText(img_to_show, ss.str(), cv::Point(5, text_size.height + 5), FONT_FACE, 1, cv::Scalar(255,0,0), FONT_THICKNESS);

    if (wait) {
        cv::destroyWindow(used_algorithms_ + " - Iteration");
        cv::imshow(used_algorithms_ + " - Result", img_to_show);
        cv::waitKey(0);
        cv::destroyWindow(used_algorithms_ + " - Result");
    }
    objects_found_.clear();
}

void ObjectDetector::drawCountourWithText(cv::Mat& img, std::vector<cv::Point2f>& countour, std::string text) {
    line(img, countour[0], countour[1], cv::Scalar(0, 255, 0), 4);
    line(img, countour[1], countour[2], cv::Scalar(0, 255, 0), 4);
    line(img, countour[2], countour[3], cv::Scalar(0, 255, 0), 4);
    line(img, countour[3], countour[0], cv::Scalar(0, 255, 0), 4);

    cv::Rect bounding_rect = cv::boundingRect(countour);
    cv::Point2f center(bounding_rect.x + bounding_rect.width / 2.0, bounding_rect.y + bounding_rect.height / 2.0);
    double max_dimen = bounding_rect.width > bounding_rect.height ? bounding_rect.width : bounding_rect.height;

    double font_scale = 1;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, FONT_FACE, font_scale, FONT_THICKNESS, &baseline);

    font_scale = max_dimen / (text_size.width * FONT_RATIO);
    text_size = cv::getTextSize(text, FONT_FACE, font_scale, FONT_THICKNESS, &baseline);
    cv::Point2f text_position = center - cv::Point2f(text_size.width / 2.0, - text_size.height / 2.0);
    cv::putText(img, text, text_position, FONT_FACE, font_scale, cv::Scalar(255,0,0), FONT_THICKNESS);
}

void ObjectDetector::drawFoundObject(cv::Mat& img, FoundObject found_object) {
    std::stringstream ss;
    ss << found_object.value_;
    drawCountourWithText(img, found_object.countour_, ss.str());
}

bool ObjectDetector::allPointsInsideCountour(std::vector<cv::Point2f> countour, std::vector<cv::Point2f> inliers) {
    for (unsigned int i = 0; i < inliers.size(); ++i) {
        if(cv::pointPolygonTest(countour, inliers[i], false) < 0) {
            return false;
        }
    }
    return true;
}

void ObjectDetector::computeAll(std::string used_algorithms, cv::FeatureDetector* detector, cv::DescriptorExtractor* extractor, cv::DescriptorMatcher* matcher) {
    used_algorithms_ = used_algorithms;
    feature_detector_ = detector;
    descriptor_extractor_ = extractor;
    descriptor_matcher_ = matcher;

    Log::instance().debug("NUMBER OF KEYPOINTS\n");
    std::stringstream ss;
    for (unsigned i = 0; i < object_library_.size(); ++i) {
        object_library_[i].compute(feature_detector_, descriptor_extractor_);

        ss << object_library_[i].getTag() << ": " << object_library_[i].getKeypoints().size() << "\n";
        Log::instance().debug(ss.str());
        ss.str("");
    }

    scene_.compute(feature_detector_, descriptor_extractor_);
    ss << "scene: " << scene_.getKeypoints().size() << "\n\n";
    Log::instance().debug(ss.str());
}