#include <iostream>
#include <iomanip>

#include "windows.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include "Log.h"
#include "ObjectDetector.h"

// For a given combination <detector, extractor, matcher>,
// allocates the respective algorithms
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
        if (matcher_name == "Bruteforce") {
            matcher = new cv::BFMatcher(cv::NORM_L2, false);
            return;
        }
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
        matcher = new cv::BFMatcher(cv::NORM_HAMMING, true);
    }
}

// Scan integer input from user betwwen a given min and max values
int getInput(std::string prompt, int min, int max) {
    std::string input;
    int option;
    while (true) {
        std::cin.clear();
        std::cout << prompt;
        std::getline(std::cin, input);
        std::stringstream sstream(input);

        if (sstream >> option && option >= min && option <= max) {
            return option;
        }
    }
}

// Scan a string input from user
std::string getInput(std::string prompt) {
    std::string input;
    std::cout << prompt;
    std::getline(std::cin, input);
    return input;
}

int main(int argc, char** argv) {
    bool testing = false;
    bool with_wait = true;

    // Invalid number of arguments
    if (argc > 3) {
        std::cout << "Usage: " << argv[0] << " [<filename>] [-test]" << "\n";
        return 1;
    }

    std::string filename = "";

    // Checks parameters: filename (string) and testing mode (-t or -test)
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-test" || std::string(argv[i]) == "-t") {
            testing = true;
        } else {
            filename = argv[i];
        }
    }

    Log& log = Log::instance();
    log.open("log.txt");

    cv::FeatureDetector* detector = NULL; 
    cv::DescriptorExtractor* extractor =NULL; 
    cv::DescriptorMatcher* matcher = NULL;

    // Combinations of feature detectors, descriptor extractor
    // and descriptor matcher
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

    // Ask user the filename if none was provided as parameter
    if (filename == "") {
        filename = getInput("Filename: ");
    }

    // If in normal mode, ask user if he wants to hide the windows with the analysis results
    if (!testing) {
        std::string answer = "";
        while (answer != "y" && answer != "Y" && answer != "n" && answer != "N") {
            std::cout << "Hide windows with results (y/n)? ";
            std::getline(std::cin, answer);
        }
        if (answer == "y") {
            with_wait = false; // User don't want to see results of image analysis
        }
    } else {
        with_wait = false; // Don't show windows with results of image analysis
    }

    ObjectDetector object_detector = ObjectDetector(filename, detector, extractor, matcher);
    object_detector.loadLibrary(true);

    if (testing) {
        // -------------------------------------------------------------
        // Test mode
        // -------------------------------------------------------------
        LARGE_INTEGER frequency; // ticks per second
        LARGE_INTEGER begin, end;
        double elapsed_time;

        std::stringstream ss;

        QueryPerformanceFrequency(&frequency);
        
        std::string used_algorithms;
        // Test all available combinations. For each one, the elapsed time is recorded.
        // Some of the inner function in this loop also record some metrics.
        for (int i = 0; i < 11; ++i) {
            used_algorithms = "Feature Detector: " + combinations[i][0] + " " +
                "Descriptor Extractor: " + combinations[i][1] + " " +
                "Descriptor Matcher: " + combinations[i][2] + "\n";
            log.debug(used_algorithms);
            getCombination(combinations[i][0], combinations[i][1], combinations[i][2],
                           detector, extractor, matcher);

            object_detector.computeAll(used_algorithms, detector, extractor, matcher);

            QueryPerformanceCounter(&begin);
            object_detector.findAllObjects(with_wait);
            QueryPerformanceCounter(&end);

            // elapsed time in milliseconds
            elapsed_time = (end.QuadPart - begin.QuadPart) * 1000.0 / frequency.QuadPart;

            ss << "Elapsed time: " << elapsed_time << " ms\n";
            log.debug(ss.str());
            ss.str("");

            delete detector;
            delete extractor;
            delete matcher;
            extractor = NULL;
            detector = NULL;
            matcher = NULL;
        }
    } else {
        // -------------------------------------------------------------
        // Normal mode
        // -------------------------------------------------------------
        int choice;
        while (true) {
            std::cout << "     Feature Detector | Descriptor Extractor | Descriptor Matcher" << "\n"
                      << "-----------------------------------------------------------" << "\n";
            for (int i = 0; i < 11; ++i) {
                std::cout << std::setw(2) << i << " | " << std::setw(16) << combinations[i][0]
                                               << " | " << std::setw(20) << combinations[i][1]
                                               << " | " << std::setw(12) << combinations[i][2] << "\n";
            }
            std::cout << "-----------------------------------------------------------" << "\n"
                      << "11 | " << std::setw(54) << "All combinations" << "\n";
            std::cout << "-----------------------------------------------------------" << "\n";

            choice = getInput("Option (-1 to exit): ", -1, 11);

            if (choice == -1) {
                break;
            }

            if (choice == 11) {
                // User wants to use all the available combinations.
                for (int i = 0; i < 11; ++i) {
                    std::string title = "Feature Detector: " + combinations[i][0] + " " +
                        "Descriptor Extractor: " + combinations[i][1] + " " +
                        "Descriptor Matcher: " + combinations[i][2] + "\n";
                    getCombination(combinations[i][0], combinations[i][1], combinations[i][2],
                        detector, extractor, matcher);

                    object_detector.computeAll(title, detector, extractor, matcher);

                    object_detector.findAllObjects(with_wait);

                    delete detector;
                    delete extractor;
                    delete matcher;
                    extractor = NULL;
                    detector = NULL;
                    matcher = NULL;
                }
            } else {
                getCombination(combinations[choice][0], combinations[choice][1], combinations[choice][2],
                    detector, extractor, matcher);

                std::string title = "Feature Detector: " + combinations[choice][0] + " " +
                    "Descriptor Extractor: " + combinations[choice][1] + " " +
                    "Descriptor Matcher: " + combinations[choice][2] + "\n";
                object_detector.computeAll(title, detector, extractor, matcher);

                object_detector.findAllObjects(with_wait);

                delete detector;
                delete extractor;
                delete matcher;
                extractor = NULL;
                detector = NULL;
                matcher = NULL;
            }
        }
    }

    log.close();

    return 0;
}