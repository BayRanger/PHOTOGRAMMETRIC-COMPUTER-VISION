#ifndef HELPER_H
#define HELPER_H


#include <opencv2/opencv.hpp>


namespace pcv3 {

/**
 * @brief Display image and catch the point marked by left mouse clicks.
 * @details Points have to be clicked in same order as in the file. Points will be in homogeneous coordinates. 
 *          The file with the 3D real world coordinates needs one line per point with x, y, z separated by spaces.
 * @param calibImg Structure containing calibration image
 * @param filenameList3DPoints Path to file that contains 3D real world coordinates.
 * @param points2D Returns points within the image (produced by this method)
 * @param points3D Returns points at the object (read from file by this method)
 */
int getPoints(const cv::Mat &calibImg, const std::string &filenameList3DPoints, std::vector<cv::Vec3f> &points2D, std::vector<cv::Vec4f> &points3D);

/// Interpretation of the internal and external parts of a projection matrix.
struct ProjectionMatrixInterpretation
{
    /// Principal distance or focal length
    float principalDistance;
    /// Skew as an angle and in degrees
    float skew;
    /// Aspect ratio of the pixels
    float aspectRatio;
    /// Location of principal point in image (pixel) coordinates
    cv::Vec2f principalPoint;
    /// Camera rotation angle 1/3
    float omega;
    /// Camera rotation angle 2/3
    float phi;
    /// Camera rotation angle 3/3
    float kappa;
    /// 3D camera location in world coordinates
    cv::Vec3f cameraLocation;
};


std::ostream &operator<<(std::ostream &stream, const ProjectionMatrixInterpretation &info);

}


#endif // HELPER_H
