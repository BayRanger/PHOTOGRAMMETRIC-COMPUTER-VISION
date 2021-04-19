//============================================================================
// Name        : Pcv1.h
// Author      : Ronny Haensch, Andreas Ley
// Version     : 2.0
// Copyright   : -
// Description : header file for first PCV assignment
//============================================================================

#ifndef PCV1_INCLUDED
#define PCV1_INCLUDED

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

namespace pcv1 {
    
// functions to be implemented
// --> please edit ONLY these functions!



/**
 * @brief Convert a 2D point from Euclidean to homogeneous coordinates
 * @param p The point to convert (in Euclidean coordinates)
 * @returns The same point in homogeneous coordinates
 */
cv::Vec3f eucl2hom_point_2D(const cv::Vec2f& p);

/**
 * @brief Convert a 2D point from homogeneous to Euclidean coordinates
 * @param p The point to convert in homogeneous coordinates 
 * @returns The same point in Euclidean coordinates
 */
cv::Vec2f hom2eucl_point_2D(const cv::Vec3f& p);

/**
 * @brief Checks if a point is on a line
 * @param point The given point in homogeneous coordinates
 * @param line The given line in homogeneous coordinates
 * @param eps The used accuracy (allowed distance to still be considered on the line)
 * @returns Returns true if the point is on the line
 */
bool isPointOnLine_2D(const cv::Vec3f& point, const cv::Vec3f& line, float eps = 1e-5f);


enum GeometryType {
    GEOM_TYPE_POINT,
    GEOM_TYPE_LINE,
};

/**
 * @brief Applies a 2D transformation to an array of points or lines
 * @param H Matrix representing the transformation
 * @param geomObjects Array of input objects, each in homogeneous coordinates
 * @param type The type of the geometric objects, point or line. All are the same type.
 * @returns Array of transformed objects.
 */
std::vector<cv::Vec3f> applyH_2D(const std::vector<cv::Vec3f>& geomObjects, const cv::Matx33f &H, GeometryType type);


/**
 * @brief Combines translation-, rotation-, and scaling-matrices to a single transformation matrix H
 * @details The returned transformation behaves as if objects were first transformed by T, then by R, and finally by S.
 * @param Translation matrix
 * @param Rotation matrix
 * @param Scaling matrix
 * @returns The combined homography
 */
cv::Matx33f getH_2D(const cv::Matx33f& T, const cv::Matx33f& R, const cv::Matx33f& S);



/**
 * @brief Generates a 2D isotropic scaling matrix S defined by scaling factor lambda
 * @param lambda The scaling parameter
 * @returns The resulting scaling matrix
 */
cv::Matx33f getScalingMatrix_2D(float lambda);

/**
 * @brief Generates a 2D rotation matrix R defined by angle phi
 * @param phi The rotation angle in degree (!)
 * @returns The resulting rotation matrix
 */
cv::Matx33f getRotationMatrix_2D(float phi);

/**
 * @brief Generates a 2D translation matrix T defined by translation (dx, dy)^T
 * @param dx The translation in x-direction
 * @param dy the translation in y-direction
 * @returns The resulting translation matrix
 */
cv::Matx33f getTranslationMatrix_2D(float dx, float dy);

/**
 * @brief Calculates the joining line between two points (in 2D)
 * @param p1 First of the two points in homogeneous coordinates
 * @param p2 Second of the two points in homogeneous coordinates
 * @returns The joining line in homogeneous coordinates
*/
cv::Vec3f getConnectingLine_2D(const cv::Vec3f& p1, const cv::Vec3f& p2);

/**
 * @brief Function loads input image, calls processing function and saves result (usually)
 * @param fname Path to input image
 */
void run(const std::string &imageFilename);



}


#endif // PCV1_INCLUDED
