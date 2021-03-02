//============================================================================
// Name        : Pcv4.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 2.0
// Copyright   : -
// Description : Estimation of Fundamental Matrix
//============================================================================

#include "Pcv4.h"

#include <random>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;


namespace pcv4 {
    
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
        switch (type) {
            case GEOM_TYPE_POINT: {
                // if objects are point multiply with the homography matrix
                for (int i = 0; i < geomObjects.size(); i++) {
                    cv::Vec3f geo_obj = H * geomObjects[i];
                    result.push_back(geo_obj);
                }

            }
                break;
            case GEOM_TYPE_LINE: {
                //if objects are lines multiply with the inverse of the homography matrix
                cv::Matx33f inv_t_H = H.inv().t();
                for (int i = 0; i < geomObjects.size(); i++) {
                    cv::Vec3f geo_obj = inv_t_H * geomObjects[i];
                    result.push_back(geo_obj);
                }

            }
                break;
            default:
                throw std::runtime_error("Unhandled geometry type!");
        }
        return result;
}


/**
 * @brief Get the conditioning matrix of given points
 * @param p The points as matrix
 * @returns The condition matrix
 */
cv::Matx33f getCondition2D(const std::vector<cv::Vec3f>& points2D)
{
  //cv::Matx33f condition_mat;
    float s_x(0), s_y(0), t_x(0), t_y(0);
    for (int i =0; i < points2D.size(); i++)
    {

        t_x += points2D[i][0];
        t_y += points2D[i][1];
    }
    float num = points2D.size();
 
    t_x = t_x/num;
    t_y = t_y/num;
    for (int i =0; i < points2D.size(); i++)
    {
        s_x += std::abs(points2D[i][0]- t_x);
        s_y += std::abs(points2D[i][1]- t_y);
     }

     s_x =s_x/num;
     s_y =s_y/num;
 
    cv::Matx33f  condition_mat {1.f/s_x,0.,-t_x/s_x,
        0.,1.f/s_y,-t_y/s_y,0.,0.,1.};
    return condition_mat;
}


/**
 * @brief Define the design matrix as needed to compute fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @returns The design matrix to be computed
 */
cv::Mat_<float> getDesignMatrix_fundamental(const std::vector<cv::Vec3f>& p1_conditioned, const std::vector<cv::Vec3f>& p2_conditioned)
{
    // TO DO !!!
    int pair_size = p1_conditioned.size();
    if (p2_conditioned.size()!=pair_size || pair_size <8)
    {
        std::cout<<"ERROR: The pair size doesn't satisfy the prerequisites"<<std::endl;
        return cv::Mat_<float>();

     }
    cv::Mat_<float> design_mat = cv::Mat_<float>::zeros(pair_size, 9);
    for (int i=0; i < p1_conditioned.size(); i++)
    {
        float x1 = p1_conditioned[i][0];
        float y1 = p1_conditioned[i][1];
        float x2 = p2_conditioned[i][0];
        float y2 = p2_conditioned[i][1];        
        //std::cout<<"HCX "<<x1<<", "<<y1<<" , "<<x2<<" , "<<x2<<" , "<<y2<<std::endl;

        design_mat(i,0) = x1*x2;
        design_mat(i,1) = y1*x2;
        design_mat(i,2) = x2;
        design_mat(i,3) = x1*y2;
        design_mat(i,4) = y1*y2;
        design_mat(i,5) = y2;
        design_mat(i,6) = x1;
        design_mat(i,7) = y1;
        design_mat(i,8) = 1.f;

    }

    return design_mat;
}


/**
 * @brief Solve homogeneous equation system by usage of SVD
 * @param A The design matrix
 * @returns The estimated fundamental matrix
 */
cv::Matx33f solve_dlt_fundamental(const cv::Mat_<float>& A)
{
    // TO DO !!!
    int A_col = A.cols;
    cv::SVD svd(A, cv::SVD::FULL_UV);
    cv::Matx33f F_mat = cv::Matx33f::zeros();
    int idx =0;
    float last_ele = svd.vt.at<float>(A_col-1 ,8);
    for (int i=0 ; i< 3;i++) // 3 is the row number of dst
    {
        for (int j=0 ; j< 3;j++)
            {
                idx = i*3+j;
                F_mat(i,j) = svd.vt.at<float>(A_col-1 ,idx)/last_ele;
            }
    }
    return F_mat;
}


/**
 * @brief Enforce rank of 2 on fundamental matrix
 * @param F The matrix to be changed
 * @return The modified fundamental matrix
 */
cv::Matx33f forceSingularity(const cv::Matx33f& F)
{
    // TO DO !!!
    Mat w,vt ,u;
    SVD::compute(F,w,u,vt);
    w.at<float>(2) = 0;
    Mat F_singular = u * Mat::diag(w) * vt;

    return F_singular;
}


/**
 * @brief Decondition a fundamental matrix that was estimated from conditioned points
 * @param T1 Conditioning matrix of set of 2D image points
 * @param T2 Conditioning matrix of set of 2D image points
 * @param F Conditioned fundamental matrix that has to be un-conditioned
 * @return Un-conditioned fundamental matrix
 */
cv::Matx33f decondition_fundamental(const cv::Matx33f& T1, const cv::Matx33f& T2, const cv::Matx33f& F)
{
    // TO DO !!!
     cv::Matx33f deconditionedF= T2.t() * F * T1;
    return deconditionedF;
}


/**
 * @brief Compute the fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @return The estimated fundamental matrix
 */
cv::Matx33f getFundamentalMatrix(const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2)
{
    // TO DO !!!
    //std::cout<<"H"<<std::endl;
    std::vector<cv::Vec3f> conditioned_p1_vec, conditioned_p2_vec;
    auto condition_p1_mat = getCondition2D(p1);
    auto condition_p2_mat = getCondition2D(p2);
    conditioned_p1_vec = applyH_2D(p1, condition_p1_mat,GEOM_TYPE_POINT);
    conditioned_p2_vec = applyH_2D(p2, condition_p2_mat,GEOM_TYPE_POINT);

    cv::Mat_<float> design_fundamental_mat = getDesignMatrix_fundamental(conditioned_p1_vec, conditioned_p2_vec);

    cv::Matx33f F_hat = solve_dlt_fundamental(design_fundamental_mat);
    F_hat =forceSingularity(F_hat);

    cv::Matx33f F_mat = decondition_fundamental(condition_p1_mat,condition_p2_mat,F_hat);

    return F_mat;
}



/**
 * @brief Calculate geometric error of estimated fundamental matrix for a single point pair
 * @details Implement the "Sampson distance"
 * @param p1		first point
 * @param p2		second point
 * @param F		fundamental matrix
 * @returns geometric error
 */
float getError(const cv::Vec3f& p1, const cv::Vec3f& p2, const cv::Matx33f& F)
{
    // TO DO !!!
    float xfx = p2.dot( F * p1);
    float denom = (F*p1)[0]*(F*p1)[0]+ (F*p1)[1]*(F*p1)[1]+ (F.t()*p2)[0]*(F.t()*p2)[0]+ (F.t()*p2)[1]*(F.t()*p2)[1];

    float result =(xfx*xfx)/denom;
    //std::cout<<"Error: "<<result<<std::endl;

    return result;
}

/**
 * @brief Calculate geometric error of estimated fundamental matrix for a set of point pairs
 * @details Implement the mean "Sampson distance"
 * @param p1		first set of points
 * @param p2		second set of points
 * @param F		fundamental matrix
 * @returns geometric error
 */
float getError(const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2, const cv::Matx33f& F)
{
    float error_sum = 0;
    for (int i =0; i<p1.size(); i++)
    {
        float error_tmp = getError(p1[i],p2[i],F);
        error_sum += error_tmp;
    }
    return error_sum/(p1.size());
}

/**
 * @brief Count the number of inliers of an estimated fundamental matrix
 * @param p1		first set of points
 * @param p2		second set of points
 * @param F		fundamental matrix
 * @param threshold Maximal "Sampson distance" to still be counted as an inlier
 * @returns		Number of inliers
 */
unsigned countInliers(const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2, const cv::Matx33f& F, float threshold)
{
    unsigned counter =0;
        for (int i =0; i<p1.size(); i++)
    {
        float error_tmp = getError(p1[i],p2[i],F);
        if (error_tmp <= threshold)
        {
            counter ++;
        }
    }
    
    return counter;
}




/**
 * @brief Estimate the fundamental matrix robustly using RANSAC
 * @details Use the number of inliers as the score
 * @param p1 first set of points
 * @param p2 second set of points
 * @param numIterations How many subsets are to be evaluated
 * @param threshold Maximal "Sampson distance" to still be counted as an inlier
 * @returns The fundamental matrix
 */
cv::Matx33f estimateFundamentalRANSAC(const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2, unsigned numIterations, float threshold)
{
    const unsigned subsetSize = 8;
    int max_inlier = 0;
    int inlier_tmp = 0;
    cv::Matx33f best_F;
    
    std::mt19937 rng;
    std::uniform_int_distribution<unsigned> uniformDist(0, p1.size()-1);
    // Draw a random point index with unsigned index = uniformDist(rng);
    for (int i =0; i< numIterations; i++)
    {
        std::vector<cv::Vec3f> sample_vec1, sample_vec2;

        for (int j =0; j< subsetSize; j++)
        {
            //discuss how to avoid duplicated elements
            unsigned idx = uniformDist(rng);
            sample_vec1.push_back(p1[idx]);
            sample_vec2.push_back(p2[idx]);

        }
        cv::Matx33f F_tmp = getFundamentalMatrix(sample_vec1, sample_vec2);
        inlier_tmp = countInliers(sample_vec1, sample_vec2, F_tmp,threshold);
        
        if (inlier_tmp > max_inlier)
        {
            max_inlier = inlier_tmp;
            best_F = F_tmp;
        }
    }
    
    return  best_F;
}




/**
 * @brief Draw points and corresponding epipolar lines into both images
 * @param img1 Structure containing first image
 * @param img2 Structure containing second image
 * @param p1 First point set (points in first image)
 * @param p2 First point set (points in second image)
 * @param F Fundamental matrix (mapping from point in img1 to lines in img2)
 */
void visualize(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2, const cv::Matx33f& F)
{
    // make a copy to not draw into the original images and destroy them
    cv::Mat img1_copy = img1.clone();
    cv::Mat img2_copy = img2.clone();

    for (int i =0 ; i <p1.size(); i++)
    {
        //set points respectively
        float x1 = p1[i][0]/p1[i][2];
        float y1 = p1[i][1]/p1[i][2];
        float x2 = p2[i][0]/p2[i][2];
        float y2 = p2[i][1]/p2[i][2];
        
        //draw points
        cv::circle(img1_copy, cv::Point2f(x1, y1), 2, cv::Scalar(0, 255, 0), 2); 
        cv::circle(img2_copy, cv::Point2f(x2, y2), 2, cv::Scalar(255, 255, 0), 2); 
        //draw Epilines
        auto epi_in_p2 = F*p1[i];    
        drawEpiLine(img2_copy,epi_in_p2[0],epi_in_p2[1],epi_in_p2[2]);
        //ax+by+c=0
        auto epi_in_p1 = F.t()*p2[i];
        drawEpiLine(img1_copy,epi_in_p1[0],epi_in_p1[1],epi_in_p1[2]);

    }
        
    // TO DO !!!
    // Compute epilines for both images and draw them with drawEpiLine() into img1_copy and img2_copy respectively
    // Use cv::circle(image, cv::Point2f(x, y), 2, cv::Scalar(0, 255, 0), 2); to draw the points.
    
    // show images
    cv::imshow("Epilines img1", img1_copy);
    cv::imshow("Epilines img2", img2_copy);
    
    cv::waitKey(0);
}



/**
 * @brief Filters the raw matches
 * @details Applies cross consistency check and ratio test (ratio of 0.75) and returns the point pairs that pass both.
 * @param rawOrbMatches Structure containing keypoints and raw matches obtained from comparing feature descriptors (see Helper.h)
 * @param p1 Points within the first image (returned in the array by this method)
 * @param p2 Points within the second image (returned in the array by this method)
 */
void filterMatches(const RawOrbMatches &rawOrbMatches, std::vector<cv::Vec3f>& p1, std::vector<cv::Vec3f>& p2)
{
    
/******* Small std::map cheat sheet ************************************

// This std::map stores pairs of ints and floats (key value pairs). Each float (value) can quickly be looked up with it's corresponding int (key).
std::map<int, float> exampleMap;
 
// Looking up an element:
int key = 5;
auto it = exampleMap.find(key);
if (it == exampleMap.end()) {
    // no entry with key 5 in the map
} else {
    float value = it->second;
    // do s.th. with the value
}

// Iteration over all elements: 
for (const auto &pair : exampleMap) {
    int key = pair.first;
    float value = pair.second;
}

**************************************************************************/

    p1.clear();
    p2.clear();
    
    const float ratio = 0.75f;

    for (const auto &pair : rawOrbMatches.matches_1_2) {
        
        // TO DO !!!
        // Skip those pairs that don't fulfill the ratio test or cross consistency check
        //should smaller thatn the ratio test
        float dist1 = pair.second.closestDistance;
        float dist2 = pair.second.secondClosestDistance;
        if (dist1/dist2 >= ratio)
        {
            continue;
        }

        unsigned key_idx = pair.second.closest;
        auto it = rawOrbMatches.matches_2_1.find(key_idx);
        if (it == rawOrbMatches.matches_2_1.end()) {
            continue;
            // no entry with key 5 in the map
        } 
        else {
            auto corre_id = it->second.closest;
            if (corre_id != pair.first)
                {       
                    continue;
                }
        }

        p1.push_back(rawOrbMatches.keypoints1[pair.first]);
        p2.push_back(rawOrbMatches.keypoints2[pair.second.closest]);
    }
}

/**
 * @brief Computes matches automatically.
 * @details Points will be in homogeneous coordinates.
 * @param img1 The first image
 * @param img2 The second image
 * @param p1 Points within the first image (returned in the array by this method)
 * @param p2 Points within the second image (returned in the array by this method)
 */
void getPointsAutomatic(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Vec3f>& p1, std::vector<cv::Vec3f>& p2)
{
    // TO DO !!!
    RawOrbMatches rawOrbMatches = extractRawOrbMatches(img1, img2);
    filterMatches(rawOrbMatches,p1,p2);
    drawMatches(img1,img2,p1,p2);
}


}
