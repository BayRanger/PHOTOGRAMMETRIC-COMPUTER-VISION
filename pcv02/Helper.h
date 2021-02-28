#ifndef HELPER_H
#define HELPER_H


#include <opencv2/opencv.hpp>


namespace pcv2 {

/**
 * @brief Displays two images and catches the point pairs marked by left mouse clicks.
 * @details Points will be in homogeneous coordinates.
 * @param baseImg The base image
 * @param attachImg The image to be attached
 * @param points_base Points within the base image (returned in the matrix by this method)
 * @param points_attach Points within the second image (returned in the matrix by this method)
 */
int getPoints(const cv::Mat &baseImg, const cv::Mat &attachImg, std::vector<cv::Vec3f> &points_base, std::vector<cv::Vec3f> &points_attach);

/**
 * @brief Stitches two images together by transforming one of them by a given homography
 * @param baseImg The base image
 * @param attachImg The image to be attached
 * @param H The homography to warp the second image
 * @return The resulting image
 */
cv::Mat stitch(const cv::Mat &baseImg, const cv::Mat &attachImg, const cv::Matx33f &H);

}


#endif // HELPER_H
