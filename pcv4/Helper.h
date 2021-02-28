#ifndef HELPER_H
#define HELPER_H


#include <opencv2/opencv.hpp>

#include <vector>
#include <map>

namespace pcv4 {

/**
 * @brief Displays two images and catches the point pairs marked by left mouse clicks.
 * @details Points will be in homogeneous coordinates.
 * @param img1 The first image
 * @param img2 The second image
 * @param p1 Points within the first image (returned in the matrix by this method)
 * @param p2 Points within the second image (returned in the matrix by this method)
 */
int getPointsManual(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Vec3f>& p1, std::vector<cv::Vec3f>& p2);

/** 
 * @brief Draws line given in homogeneous representation into image
 * @param img the image to draw into
 * @param a The line parameters
 * @param b The line parameters
 * @param c The line parameters
 */
void drawEpiLine(cv::Mat& img, double a, double b, double c);


/**
 * @brief Concatenates two images and draws the matches between them as lines
 * @param img1 First image
 * @param img2 Second image
 * @param p1 List of keypoints in first image
 * @param p2 List of corresponding keypoints in second image
 * @return Image showing matches
 */
cv::Mat drawMatches(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Vec3f>& p1, std::vector<cv::Vec3f>& p2);


struct RawOrbMatches {
    /// Location of keypoints in first image
    std::vector<cv::Vec3f> keypoints1;
    /// Location of keypoints in second image
    std::vector<cv::Vec3f> keypoints2;
    
    struct Match {
        /// Index of the closest match
        unsigned closest;
        /// Feature distance of closest match
        float closestDistance;
        /// Index of the second closest match
        unsigned secondClosest;
        /// Feature distance of second closest match
        float secondClosestDistance;
    };
    
    /// @brief For each keypoint in first images which keypoints are similar in second image
    /// @details The key is the index of the keypoint in the first image, the value is a Match struct containing the indices and distances of the closest matching keypoints in the second image.
    std::map<unsigned, Match> matches_1_2;
    
    /// @brief For each keypoint in second images which keypoints are similar in first image
    /// @details So like matches_1_2, just with the two images reversed.
    std::map<unsigned, Match> matches_2_1;
};


/**
 * @brief Computes and matches ORB feature points in two images
 * @param img1 First image
 * @param img2 Second image
 * @return Structure containing keypoint locations and matches
 */
RawOrbMatches extractRawOrbMatches(const cv::Mat& img1, const cv::Mat& img2);


}


#endif // HELPER_H
