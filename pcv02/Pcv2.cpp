//============================================================================
// Name        : Pcv2test.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include "Pcv2.h"
#include <cmath>

namespace pcv2 {


/**
 * @brief get the conditioning matrix of given points
 * @param the points as matrix
 * @returns the condition matrix (already allocated)
 */
cv::Matx33f getCondition2D(const std::vector<cv::Vec3f> &points)
{
    // TO DO !!!

    //cv::Matx33f condition_mat;
    float s_x(0), s_y(0), t_x(0), t_y(0);
    for (int i =0; i < points.size(); i++)
    {
        t_x += points[i][0];
        t_y += points[i][1];
    }
    float num = points.size();
    t_x = t_x/num;
    t_y = t_y/num;
    for (int i =0; i < points.size(); i++)
    {
        s_x += std::abs(points[i][0]- t_x);
        s_y += std::abs(points[i][1]- t_y);
    }

     s_x =s_x/num;
     s_y =s_y/num;
 
    cv::Matx33f  condition_mat {1.f/s_x,0.,-t_x/s_x,
        0.,1.f/s_y,-t_y/s_y,0.,0.,1.};
    return condition_mat;
}


/**
 * @brief define the design matrix as needed to compute 2D-homography
 * @param conditioned_base first set of conditioned points x' --> x' = H * x
 * @param conditioned_attach second set of conditioned points x --> x' = H * x
 * @returns the design matrix to be computed
 */
cv::Mat_<float> getDesignMatrix_homography2D(const std::vector<cv::Vec3f> &conditioned_base, const std::vector<cv::Vec3f> &conditioned_attach)
{
    int pair_size = conditioned_base.size();

    cv::Mat_<float> dst = cv::Mat_<float>::zeros(pair_size * 2, 9);
    for (int pair_num =0;  pair_num < pair_size; pair_num ++ )
    {
        float u_ = conditioned_base[pair_num][0];//
        float v_ = conditioned_base[pair_num][1];//
        float w_ = conditioned_base[pair_num][2];//
        dst(pair_num*2 +0,0) = - w_ * conditioned_attach[pair_num][0];
        dst(pair_num*2 +0,1) = - w_ * conditioned_attach[pair_num][1];
        dst(pair_num*2 +0,2) = - w_ * conditioned_attach[pair_num][2];
        dst(pair_num*2 +0,6) =   u_ * conditioned_attach[pair_num][0];
        dst(pair_num*2 +0,7) =   u_ * conditioned_attach[pair_num][1];
        dst(pair_num*2 +0,8) =   u_ * conditioned_attach[pair_num][2];
        dst(pair_num*2 +1,3) = - w_ * conditioned_attach[pair_num][0];
        dst(pair_num*2 +1,4) = - w_ * conditioned_attach[pair_num][1];
        dst(pair_num*2 +1,5) = - w_ * conditioned_attach[pair_num][2];
        dst(pair_num*2 +1,6) =   v_ * conditioned_attach[pair_num][0];
        dst(pair_num*2 +1,7) =   v_ * conditioned_attach[pair_num][1];
        dst(pair_num*2 +1,8) =   v_ * conditioned_attach[pair_num][2];
    }


    return dst;
}

/**
 * @brief solve homogeneous equation system by usage of SVD
 * @param A the design matrix
 * @returns solution of the homogeneous equation system
 */
cv::Matx33f solve_dlt_homography2D(const cv::Mat_<float> &A)
{
    int A_row = A.rows;
    int A_col = A.cols;
    cv::SVD svd(A, cv::SVD::FULL_UV);
    cv::Matx33f dst = cv::Matx33f::eye();
    int idx =0;

    for (int i=0 ; i< 3;i++) // 3 is the dimension of dst
    {
        for (int j=0 ; j< 3 ;j++)
            {
                idx = i*3+j;

                dst(i,j) = svd.vt.at<float>(A_col-1 ,idx);
            }

    }
    int sgn = (dst(2,2)>0 )? 1 : -1;
    dst *= sgn;
    return dst;

}
/**
 * @brief decondition a homography that was estimated from conditioned point clouds
 * @param T_base conditioning matrix T' of first set of points x'
 * @param T_attach conditioning matrix T of second set of points x
 * @param H conditioned homography that has to be un-conditioned (in-place)
 */
cv::Matx33f decondition_homography2D(const cv::Matx33f &T_base, const cv::Matx33f &T_attach, const cv::Matx33f &H) 
{
    // TO DO !!!
    cv::Matx33f dst;
    dst = T_base.inv() *H * T_attach;

    return dst;
}


/**
 * @brief compute the homography
 * @param base first set of points x'
 * @param attach second set of points x
 * @returns homography H, so that x' = Hx
 */
cv::Matx33f homography2D(const std::vector<cv::Vec3f> &base, const std::vector<cv::Vec3f> &attach)
{
    // TO DO !!!
    cv::Matx33f base_condition_mat = getCondition2D(base);
    cv::Matx33f attach_condition_mat = getCondition2D(attach);
    std::vector<cv::Vec3f> base_conditioned_vec, attach_conditioned_vec;

    base_conditioned_vec = applyH_2D(base , base_condition_mat, GEOM_TYPE_POINT);
    attach_conditioned_vec = applyH_2D(attach , attach_condition_mat, GEOM_TYPE_POINT);
    
    cv::Mat_<float> design_mat = getDesignMatrix_homography2D(base_conditioned_vec, attach_conditioned_vec);

    cv::Matx33f homography_mat =solve_dlt_homography2D(design_mat );
    cv::Matx33f dst = decondition_homography2D(base_condition_mat, attach_condition_mat,homography_mat ) ;

    return dst;
}



// Functions from exercise 1
// Reuse your solutions from the last exercise here

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
             for (int i = 0; i < geomObjects.size(); i++) {
                    cv::Vec3f geo_obj = H * geomObjects[i];
                    result.push_back(geo_obj);
                }

        } break;
        case GEOM_TYPE_LINE: {
              cv::Matx33f inv_t_H = H.inv().t();
                for (int i = 0; i < geomObjects.size(); i++) {
                    cv::Vec3f geo_obj = inv_t_H * geomObjects[i];
                    result.push_back(geo_obj);
                }

        } break;

        default:
            throw std::runtime_error("Unhandled geometry type!");
    }
    return result;
}


/**
 * @brief Convert a 2D point from Euclidean to homogeneous coordinates
 * @param p The point to convert (in Euclidean coordinates)
 * @returns The same point in homogeneous coordinates
 */
cv::Vec3f eucl2hom_point_2D(const cv::Vec2f& p)
{
   // add 1 as the homogeneous component
    cv::Vec3f hom_point{p[0], p[1], 1};
    return hom_point;
}
}

