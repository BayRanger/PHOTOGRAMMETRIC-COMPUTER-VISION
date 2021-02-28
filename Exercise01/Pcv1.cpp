//============================================================================
// Name        : Pcv1.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include "Pcv1.h"

#include <stdexcept>

namespace pcv1 {

    
/**
 * @brief Convert a 2D point from Euclidean to homogeneous coordinates
 * @param p The point to convert (in Euclidean coordinates)
 * @returns The same point in homogeneous coordinates
 */
cv::Vec3f eucl2hom_point_2D(const cv::Vec2f& p)
{
    // TO DO !!!
    return cv::Vec3f();
}

/**
 * @brief Convert a 2D point from homogeneous to Euclidean coordinates
 * @param p The point to convert in homogeneous coordinates 
 * @returns The same point in Euclidean coordinates
 */
cv::Vec2f hom2eucl_point_2D(const cv::Vec3f& p)
{
    // TO DO !!!
    return cv::Vec2f();
}
    
    
/**
 * @brief Calculates the joining line between two points (in 2D)
 * @param p1 First of the two points in homogeneous coordinates
 * @param p2 Second of the two points in homogeneous coordinates
 * @returns The joining line in homogeneous coordinates
*/
cv::Vec3f getConnectingLine_2D(const cv::Vec3f& p1, const cv::Vec3f& p2)
{
    // TO DO !!!
    return cv::Vec3f();
}

/**
 * @brief Generates a 2D translation matrix T defined by translation (dx, dy)^T
 * @param dx The translation in x-direction
 * @param dy the translation in y-direction
 * @returns The resulting translation matrix
 */
cv::Matx33f getTranslationMatrix_2D(float dx, float dy)
{
    // TO DO !!!
    return cv::Matx33f();
}

/**
 * @brief Generates a 2D rotation matrix R defined by angle phi
 * @param phi The rotation angle in degree (!)
 * @returns The resulting rotation matrix
 */
cv::Matx33f getRotationMatrix_2D(float phi)
{
    // TO DO !!!
    return cv::Matx33f();
}

/**
 * @brief Generates a 2D isotropic scaling matrix S defined by scaling factor lambda
 * @param lambda The scaling parameter
 * @returns The resulting scaling matrix
 */
cv::Matx33f getScalingMatrix_2D(float lambda)
{
    // TO DO !!!
    return cv::Matx33f();
}

/**
 * @brief Combines translation-, rotation-, and scaling-matrices to a single transformation matrix H
 * @details The returned transformation behaves as if objects were first transformed by T, then by R, and finally by S.
 * @param Translation matrix
 * @param Rotation matrix
 * @param Scaling matrix
 * @returns The combined homography
 */
cv::Matx33f getH_2D(const cv::Matx33f& T, const cv::Matx33f& R, const cv::Matx33f& S)
{
    // TO DO !!!
    return cv::Matx33f();
}

/**
 * @brief Applies a 2D transformation to an array of points or lines
 * @param H Matrix representing the transformation
 * @param geomObjects Array of input objects, each in homogeneous coordinates
 * @param type The type of the geometric objects, point or line. All are the same type.
 * @returns Array of transformed objects.
 */
std::vector<cv::Vec3f> applyH_2D(const std::vector<cv::Vec3f>& geomObjects, const cv::Matx33f &H, GeometryType type)
{
    std::vector<cv::Vec3f> result;
    
    /******* Small std::vector cheat sheet ************************************/
    /*
     *   Number of elements in vector:                 a.size()
     *   Access i-th element (reading or writing):     a[i]
     *   Resize array:                                 a.resize(count);
     *   Append an element to an array:                a.push_back(element);
     *     \-> preallocate memory for e.g. push_back:  a.reserve(count);
     */
    /**************************************************************************/

    // TO DO !!!

    switch (type) {
        case GEOM_TYPE_POINT: {
            // TO DO !!!
        } break;
        case GEOM_TYPE_LINE: {
            // TO DO !!!
        } break;
        default:
            throw std::runtime_error("Unhandled geometry type!");
    }
    return result;
}

/**
 * @brief Checks if a point is on a line
 * @param point The given point in homogeneous coordinates
 * @param line The given line in homogeneous coordinates
 * @param eps The used accuracy (allowed distance to still be considered on the line)
 * @returns Returns true if the point is on the line
 */
bool isPointOnLine_2D(const cv::Vec3f& point, const cv::Vec3f& line, float eps)
{
    // TO DO !!!
    return false;
}



/**
 * @brief Function loads input image, calls processing function and saves result (usually)
 * @param fname Path to input image
 */
void run(const std::string &fname){

    // window names
    std::string win1 = "Image";

    // load image as gray-scale, path in argv[1]
    std::cout << "Load image: start" << std::endl;
    cv::Mat inputImage;
    
    // TO DO !!!
    // inputimage = ???
    
    if (!inputImage.data){
        std::cout << "ERROR: image could not be loaded from " << fname << std::endl;
        std::cout << "Press enter to continue..." << std::endl;
        std::cin.get();
    }else
        std::cout << "Load image: done ( " << inputImage.rows << " x " << inputImage.cols << " )" << std::endl;
    
    // show input image
    cv::namedWindow( win1.c_str(), cv::WINDOW_AUTOSIZE );
    cv::imshow( win1.c_str(), inputImage );
    cv::waitKey(0);

    // the two given points as OpenCV matrices
    cv::Vec2f x(2.0f, 3.0f);
    cv::Vec2f y(-4.0f, 5.0f);

    // same points in homogeneous coordinates
    cv::Vec3f v1, v2;
    // TO DO !!!
    // define v1 as homogeneous version of x
    // define v2 as homogeneous version of y
    
    // print points
    std::cout << "point 1: " << v1.t() << "^T" << std::endl;
    std::cout << "point 2: " << v2.t() << "^T" << std::endl;
    std::cout << std::endl;
    
    // connecting line between those points in homogeneous coordinates
    cv::Vec3f line = getConnectingLine_2D(v1, v2);
    
    // print line
    std::cout << "joining line: " << line << "^T" << std::endl;
    std::cout << std::endl;    
    
    // the parameters of the transformation
    int dx = 6;				// translation in x
    int dy = -7;			// translation in y
    float phi = 15;		// rotation angle in degree
    float lambda = 8;		// scaling factor

    // matrices for transformation
    // calculate translation matrix
    cv::Matx33f T = getTranslationMatrix_2D(dx, dy);
    // calculate rotation matrix
    cv::Matx33f R = getRotationMatrix_2D(phi);
    // calculate scale matrix
    cv::Matx33f S = getScalingMatrix_2D(lambda);
    // combine individual transformations to a homography
    cv::Matx33f H = getH_2D(T, R, S);
    
    // print calculated matrices
    std::cout << "Translation matrix: " << std::endl;
    std::cout << T << std::endl;
    std::cout << std::endl;
    std::cout << "Rotation matrix: " << std::endl;
    std::cout << R << std::endl;
    std::cout << std::endl;
    std::cout << "Scaling matrix: " << std::endl;
    std::cout << S << std::endl;
    std::cout << std::endl;
    std::cout << "Homography: " << std::endl;
    std::cout << H << std::endl;
    std::cout << std::endl;

    // transform first point x (and print it)
    cv::Vec3f v1_new = applyH_2D({v1}, H, GEOM_TYPE_POINT)[0];
    std::cout << "new point 1: " << v1_new << "^T" << std::endl;
    std::cout << "new point 1 (eucl): " << hom2eucl_point_2D(v1_new).t() << "^T" << std::endl;
    // transform second point y (and print it)
    cv::Vec3f v2_new = applyH_2D({v2}, H, GEOM_TYPE_POINT)[0];
    std::cout << "new point 2: " << v2_new << "^T" << std::endl;
    std::cout << "new point 2 (eucl): " << hom2eucl_point_2D(v2_new).t() << "^T" << std::endl;
    std::cout << std::endl;
    // transform joining line (and print it)
    cv::Vec3f line_new = applyH_2D({line}, H, GEOM_TYPE_LINE)[0];
    std::cout << "new line: " << line_new << "^T" << std::endl;
    std::cout << std::endl;

    // check if transformed points are still on transformed line
    bool xOnLine = isPointOnLine_2D(v1_new, line_new);
    bool yOnLine = isPointOnLine_2D(v2_new, line_new);
    if (xOnLine)
        std::cout << "first point lies still on the line *yay*" << std::endl;
    else
        std::cout << "first point does not lie on the line *oh oh*" << std::endl;

    if (yOnLine)
        std::cout << "second point lies still on the line *yay*" << std::endl;
    else
        std::cout << "second point does not lie on the line *oh oh*" << std::endl;

}


}
