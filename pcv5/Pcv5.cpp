//============================================================================
// Name        : Pcv5.cpp
// Author      : Andreas Ley
// Version     : 1.0
// Copyright   : -
// Description : Bundle Adjustment
//============================================================================

#include "Pcv5.h"

#include <random>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

namespace pcv5 {


    
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
 * @brief Applies a 3D transformation to an array of points
 * @param H Matrix representing the transformation
 * @param points Array of input points, each in homogeneous coordinates
 * @returns Array of transformed objects.
 */
std::vector<cv::Vec4f> applyH_3D_points(const std::vector<cv::Vec4f>& points, const cv::Matx44f &H)
{
     std::vector<cv::Vec4f> result;
 
    for(int i = 0; i < points.size(); i++){

        cv::Vec4f geopoint = points[i];
        cv::Vec4f tmp;
        tmp[0] = geopoint[0] * H(0,0) + geopoint[1] * H(0,1) + geopoint[2]*H(0,2)+ geopoint[3]* H(0,3);
        tmp[1] = geopoint[0] * H(1,0) + geopoint[1] * H(1,1) + geopoint[2]*H(1,2)+ geopoint[3]* H(1,3);
        tmp[2] = geopoint[0] * H(2,0) + geopoint[1] * H(2,1) + geopoint[2]*H(2,2)+ geopoint[3]* H(2,3);
        tmp[3] = geopoint[0] * H(3,0) + geopoint[1] * H(3,1) + geopoint[2]*H(3,2)+ geopoint[3]* H(3,3);

        result.push_back(tmp);

    }
    

    return result;
}

/**
 * @brief Get the conditioning matrix of given points
 * @param p The points as matrix
 * @returns The condition matrix 
 */
cv::Matx44f getCondition3D(const std::vector<cv::Vec4f>& points)
{
    float s_x(0), s_y(0), s_z(0), t_x(0), t_y(0), t_z(0);
    for (int i =0; i < points.size(); i++)
    {

        t_x += points[i][0];
        t_y += points[i][1];
        t_z += points[i][2];
    }
    float num = points.size();
 
    t_x = t_x/num;
    t_y = t_y/num;
    t_z = t_z/num;

    for (int i =0; i < points.size(); i++)
    {
        s_x += std::abs(points[i][0]- t_x);
        s_y += std::abs(points[i][1]- t_y);
        s_z += std::abs(points[i][2]- t_z);
     }

     s_x =s_x/num;
     s_y =s_y/num;
     s_z =s_z/num;
 
    cv::Matx44f  condition_mat {1.f/s_x,0.,0,-t_x/s_x,
        0.,1.f/s_y,0.0,-t_y/s_y,
        0.,0.,1.f/s_z, -t_z/s_z,
        0.0, 0.0, 0.0, 1.0};
    return condition_mat;
}






/**
 * @brief Define the design matrix as needed to compute projection matrix
 * @param points2D Set of 2D points within the image
 * @param points3D Set of 3D points at the object
 * @returns The design matrix to be computed
 */
cv::Mat_<float> getDesignMatrix_camera(const std::vector<cv::Vec3f>& points2D, const std::vector<cv::Vec4f>& points3D)
{
    int pair_size = points2D.size();

    cv::Mat_<float> dst = cv::Mat_<float>::zeros(pair_size * 2, 12);

    for (int pair_num =0;  pair_num < pair_size; pair_num ++ )
    {
        float u_ = points2D[pair_num][0];//
        float v_ = points2D[pair_num][1];//
        float w_ = points2D[pair_num][2];//
        dst.at<float>(pair_num*2 +0,0) = - w_ * points3D[pair_num][0];
        dst.at<float>(pair_num*2 +0,1) = - w_ * points3D[pair_num][1];
        dst.at<float>(pair_num*2 +0,2) = - w_ * points3D[pair_num][2];
        dst.at<float>(pair_num*2 +0,3) = - w_ * points3D[pair_num][3];

        dst.at<float>(pair_num*2 +0, 8) =  u_ * points3D[pair_num][0];
        dst.at<float>(pair_num*2 +0, 9) =  u_ * points3D[pair_num][1];
        dst.at<float>(pair_num*2 +0,10) =  u_ * points3D[pair_num][2];
        dst.at<float>(pair_num*2 +0,11) =  u_ * points3D[pair_num][3];

        dst.at<float>(pair_num*2 +1,4) = - w_ * points3D[pair_num][0];
        dst.at<float>(pair_num*2 +1,5) = - w_ * points3D[pair_num][1];
        dst.at<float>(pair_num*2 +1,6) = - w_ * points3D[pair_num][2];
        dst.at<float>(pair_num*2 +1,7) = - w_ * points3D[pair_num][3];

        dst.at<float>(pair_num*2 +1,8)  =  v_ * points3D[pair_num][0];
        dst.at<float>(pair_num*2 +1,9)  =  v_ * points3D[pair_num][1];
        dst.at<float>(pair_num*2 +1,10) =  v_ * points3D[pair_num][2];
        dst.at<float>(pair_num*2 +1,11) =  v_ * points3D[pair_num][3];

    }

    return dst;
}

/**
 * @brief Solve homogeneous equation system by usage of SVD
 * @param A The design matrix
 * @returns The estimated projection matrix
 */
cv::Matx34f solve_dlt_camera(const cv::Mat_<float>& A)
{
    int A_row = A.rows;
    int A_col = A.cols;
    cv::SVD svd(A, cv::SVD::FULL_UV);
    cv::Matx34f dst = cv::Matx34f::zeros();
    int idx =0;

    for (int i=0 ; i< 3;i++) // 3 is the row number of dst
    {
        for (int j=0 ; j< 4;j++)
            {
                idx = i*4+j;

                dst(i,j) = svd.vt.at<float>(A_col-1 ,idx);
            }

    }
    float last_ele= dst(2,3);
    //to normalize it 
    if ((last_ele)==0 )
    {
        return dst;
    }
    else
    {
        dst = dst * (1.f/last_ele);
        return dst;
    }

 
}

/**
 * @brief Decondition a projection matrix that was estimated from conditioned point clouds
 * @param T_2D Conditioning matrix of set of 2D image points
 * @param T_3D Conditioning matrix of set of 3D object points
 * @param P Conditioned projection matrix that has to be un-conditioned (in-place)
 */
cv::Matx34f decondition_camera(const cv::Matx33f& T_2D, const cv::Matx44f& T_3D, const cv::Matx34f& P)
{
    cv::Matx34f dst = T_2D.inv() *P * T_3D;
    return dst;

}

/**
 * @brief Estimate projection matrix
 * @param points2D Set of 2D points within the image
 * @param points3D Set of 3D points at the object
 * @returns The projection matrix to be computed
 */
cv::Matx34f calibrate(const std::vector<cv::Vec3f>& points2D, const std::vector<cv::Vec4f>& points3D)
{
    cv::Matx33f camera_condition_mat = getCondition2D(points2D);
    cv::Matx44f obj_condition_mat = getCondition3D(points3D);
    std::vector<cv::Vec3f> img_conditioned_vec;
    std::vector<cv::Vec4f> obj_conditioned_vec;
    img_conditioned_vec = applyH_2D(points2D, camera_condition_mat,GEOM_TYPE_POINT);
    obj_conditioned_vec = applyH_3D_points(points3D, obj_condition_mat);
    cv::Mat_<float> design_mat = getDesignMatrix_camera(img_conditioned_vec, obj_conditioned_vec);
    cv::Matx34f esti_proj_mat =solve_dlt_camera(design_mat );
    cv::Matx34f projection_mat = decondition_camera(camera_condition_mat, obj_condition_mat, esti_proj_mat);
 


    return projection_mat;
}

/**
 * @brief Extract and prints information about interior and exterior orientation from camera
 * @param P The 3x4 projection matrix
 * @param K Matrix for returning the computed internal calibration
 * @param R Matrix for returning the computed rotation
 * @param info Structure for returning the interpretation such as principal distance
 */
void interprete(const cv::Matx34f &P, cv::Matx33f &K, cv::Matx33f &R, ProjectionMatrixInterpretation &info)
{
    cv::Matx33f R_mat, Q_mat,M_mat;
    for (int i =0; i < 3; i++)
    {
        for (int j =0; j < 3; j++)
        {
            M_mat(i,j) = P(i,j); 
        }
    }
 

    cv::RQDecomp3x3(M_mat,R_mat,Q_mat);//,omega_degree, phi_degree, kappa_degree);
    //TODO: get the info from R & Q
    // Additional constraints
    //check if the diagonal element is positive
    for (int i=0; i<3; i++) {
        {
            if ( R(i,i)<0 )
            {
                for (int j=0; j<3; j++) {
                    R_mat(j,i) = R_mat(j,i)* -1;
                    Q_mat(i,j) = Q_mat(i,j)* -1; 
                    }
            }
       }
    }

    K = R_mat * (1.0/R_mat(2,2));
    R = Q_mat;
    info.omega = atan2(-R(2,1),R(2,2))*180/M_PI;
    info.phi = atan2(R(2,0),std::sqrt(R(2,1)*R(2,1)+ R(2,2)* R(2,2)))*180/M_PI;
    info.kappa = atan2(-R(1,0),R(0,0))*180/M_PI;

    // Principal distance or focal length

    info.principalDistance = K(0,0);  // use average?

    // Skew as an angle and in degrees
    info.skew =  90 - atan2(-K(0,1),K(0,0))*180/CV_PI;
        
     // Aspect ratio of the pixels
    info.aspectRatio = K(1,1) /K(0,0);


    info.principalPoint(0) = K(0,2);
    info.principalPoint(1) = K(1,2);
    cv::Vec3f T_vec(P(0,3),P(1,3),P(2,3));
    cv::Vec3f C_vec = - M_mat.inv() * T_vec;   


    // 3D camera location in world coordinates
    info.cameraLocation(0) = C_vec[0];
    info.cameraLocation(1) = C_vec[1];
    info.cameraLocation(2) = C_vec[2];
 
}





/**
 * @brief Define the design matrix as needed to compute fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @returns The design matrix to be computed
 */
cv::Mat_<float> getDesignMatrix_fundamental(const std::vector<cv::Vec3f>& p1_conditioned, const std::vector<cv::Vec3f>& p2_conditioned)
{
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
    Mat w,vt ,u;
    SVD::compute(F,w,u,vt);
    w.at<float>(2) = 0;
    //std::cout<<"HCX "<<w<<u<<vt<<std::endl;
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
     cv::Matx33f deconditionedF= T2.t() * F * T1;
    return deconditionedF;
}


/**
 * @brief Compute the fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @returns	the estimated fundamental matrix
 */
cv::Matx33f getFundamentalMatrix(const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2)
{
    std::vector<cv::Vec3f> conditioned_p1_vec, conditioned_p2_vec;
        //std::cout<<"H"<<std::endl;


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
    float xfx = p2.dot( F * p1);
    float denom = (F*p1)[0]*(F*p1)[0]+ (F*p1)[1]*(F*p1)[1]+ (F.t()*p2)[0]*(F.t()*p2)[0]+ (F.t()*p2)[1]*(F.t()*p2)[1];
    float result =(xfx*xfx)/denom;
 
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


int cameraPoseScore(const std::vector<cv::Vec4f>& points, const cv::Matx34f& P2 )
{
    cv::Mat H = cv::Mat::eye(4, 4, CV_32F);
    H(cv::Range(0, 3), cv::Range(0, 4)) = (cv::Mat)P2 ;
    int score =0;
    for  (int i =0; i < points.size(); i++)
    {
        cv::Mat another_point = H* cv::Mat(points[i]);

        if (points[i][2]/points[i][3]>0 && (another_point.at<float>(0,2)/another_point.at<float>(0,3)>0))
        {
            score++;
        }
    }

    return score;
    
}








/**
 * @brief Computes the relative pose of two cameras given a list of point pairs and the camera's internal calibration.
 * @details The first camera is assumed to be in the origin, so only the external calibration of the second camera is computed. The point pairs are assumed to contain no outliers.
 * @param p1 Points in first image
 * @param p2 Points in second image
 * @param K Internal calibration matrix
 * @returns External calibration matrix of second camera
- Estimates pose of second camera if first is in the origin
- Computes essential matrix from points and K
- Computes four possible camera matrices
- Chooses the one with most points in front of both cameras
- Points can be assumed to be outlier free (no RANSAC needed)
 */
cv::Matx44f computeCameraPose(const cv::Matx33f &K, const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2)
{

    std::vector<cv::Vec3f> p1_k,p2_k;
    //derive F
    for (int i = 0; i<p1.size(); i++)
    {
        p1_k.push_back(K.inv()*p1[i]);
        p2_k.push_back(K.inv()*p2[i]);
    }
    cv::Matx33f F_mat = getFundamentalMatrix(p1_k,p2_k);
    //derive E
    cv::Matx33f E_mat = K.t() * F_mat * K;
     //SVD of E
    Mat w,vt ,u;
    SVD::compute( E_mat,w,u,vt);
    if (cv::determinant(u)<0)
    {
        u = u*-1;
    }

    if(cv::determinant(vt) <0)
    {
        vt = vt*-1;
    }
    cv::Matx33f W_mat(0,-1,0,1,0,0,0,0,1);

    // get t
    Mat t = Mat::zeros(3, 1, CV_32FC1);		//B in .pdf
 
    float t1 = u.at<float>(0,2);
    float t2 = u.at<float>(1,2);
    float t3 = u.at<float>(2,2);
    t.at<float>(0,0) = t1; 
    t.at<float>(0,1) = t2;
    t.at<float>(0,2) = t3; 
    //cv::Matx33f t_matx(0, -t3, t2, t3, 0, -t1, -t2, t1, 0);

    //get R
    cv::Mat R_mat =  u * (cv::Mat)W_mat * vt;
    cv::Mat R_mat_T =   u * (cv::Mat)W_mat.t() * vt;

    cv::Matx34f P1(1,0,0,0, 0,1,0,0, 0,0,1,0);
//t, R
    cv::Matx34f P2_1(R_mat.at<float>(0,0),R_mat.at<float>(0,1),R_mat.at<float>(0,2),t1, 
        R_mat.at<float>(1,0),R_mat.at<float>(1,1),R_mat.at<float>(1,2), t2,
        R_mat.at<float>(2,0),R_mat.at<float>(2,1), R_mat.at<float>(2,2), t3);
    std::vector<cv::Vec4f> target_point_1 = linearTriangulation( K*P1, K*P2_1, p1, p2);
    
    int best_choice = 1;
    int best_score = cameraPoseScore(target_point_1,P2_1);


//-t, R
    cv::Matx34f P2_2(R_mat.at<float>(0,0),R_mat.at<float>(0,1),R_mat.at<float>(0,2),-t1, 
        R_mat.at<float>(1,0),R_mat.at<float>(1,1),R_mat.at<float>(1,2), -t2,
        R_mat.at<float>(2,0),R_mat.at<float>(2,1), R_mat.at<float>(2,2), -t3);
    std::vector<cv::Vec4f> target_point_2 = linearTriangulation( K*P1, K*P2_2, p1, p2);
    if (cameraPoseScore(target_point_2,P2_2) > best_score)
    {
        best_choice = 2;
        best_score = cameraPoseScore(target_point_2,P2_2);
    }

//t, R_T
    cv::Matx34f P2_3(R_mat_T.at<float>(0,0),R_mat_T.at<float>(0,1),R_mat_T.at<float>(0,2),t1, 
        R_mat_T.at<float>(1,0),R_mat_T.at<float>(1,1),R_mat_T.at<float>(1,2), t2,
        R_mat_T.at<float>(2,0),R_mat_T.at<float>(2,1), R_mat_T.at<float>(2,2), t3);
    std::vector<cv::Vec4f> target_point_3 = linearTriangulation( K*P1, K*P2_3, p1, p2);
    if (cameraPoseScore(target_point_3,P2_3) > best_score)
    {
        best_choice = 3;
        best_score = cameraPoseScore(target_point_3, P2_3);

    }

//-t, R
    cv::Matx34f P2_4(R_mat_T.at<float>(0,0),R_mat_T.at<float>(0,1),R_mat_T.at<float>(0,2),-t1, 
        R_mat_T.at<float>(1,0),R_mat_T.at<float>(1,1),R_mat_T.at<float>(1,2), -t2,
        R_mat_T.at<float>(2,0),R_mat_T.at<float>(2,1), R_mat_T.at<float>(2,2), -t3);
    std::vector<cv::Vec4f> target_point_4 = linearTriangulation( K*P1, K*P2_4, p1, p2);
    if (cameraPoseScore(target_point_4,P2_4) > best_score)
    {
        best_choice = 4;
        best_score = cameraPoseScore(target_point_4,P2_4);
    }


    if(best_choice ==1)
        {
            cv::Matx44f camera_pose(R_mat.at<float>(0,0),R_mat.at<float>(0,1),R_mat.at<float>(0,2),t1, 
            R_mat.at<float>(1,0),R_mat.at<float>(1,1),R_mat.at<float>(1,2), t2,
            R_mat.at<float>(2,0),R_mat.at<float>(2,1), R_mat.at<float>(2,2), t3,
            0.f,0.f,0.f,1.f);
            return camera_pose;

        }
 
    if(best_choice ==2)
        {
            cv::Matx44f camera_pose(R_mat.at<float>(0,0),R_mat.at<float>(0,1),R_mat.at<float>(0,2),-t1, 
            R_mat.at<float>(1,0),R_mat.at<float>(1,1),R_mat.at<float>(1,2), -t2,
            R_mat.at<float>(2,0),R_mat.at<float>(2,1), R_mat.at<float>(2,2), -t3,
            0,0,0,1);
            return camera_pose;

        }
 
    if(best_choice ==3)
        {
            cv::Matx44f camera_pose(R_mat_T.at<float>(0,0),R_mat_T.at<float>(0,1),R_mat_T.at<float>(0,2),t1, 
            R_mat_T.at<float>(1,0),R_mat_T.at<float>(1,1),R_mat_T.at<float>(1,2), t2,
            R_mat_T.at<float>(2,0),R_mat_T.at<float>(2,1), R_mat_T.at<float>(2,2), t3, 
            0.f, 0.f, 0.f, 1.f);
            return camera_pose;

        }
    
    if(best_choice ==4)
        {
            cv::Matx44f camera_pose(R_mat_T.at<float>(0,0),R_mat_T.at<float>(0,1),R_mat_T.at<float>(0,2),-t1, 
            R_mat_T.at<float>(1,0),R_mat_T.at<float>(1,1),R_mat_T.at<float>(1,2), -t2,
            R_mat_T.at<float>(2,0),R_mat_T.at<float>(2,1), R_mat_T.at<float>(2,2), -t3,
            0.f,0.f,0.f,1.f);
            return camera_pose;

        }
 
 }








/**
 * @brief Estimate the fundamental matrix robustly using RANSAC
 * @param p1 first set of points
 * @param p2 second set of points
 * @param numIterations How many subsets are to be evaluated
 * @returns The fundamental matrix
 */
cv::Matx34f estimateProjectionRANSAC(const std::vector<cv::Vec3f>& points2D, const std::vector<cv::Vec4f>& points3D, unsigned numIterations, float threshold)
{
    const unsigned subsetSize = 6;

    std::mt19937 rng;
    std::uniform_int_distribution<unsigned> uniformDist(0, points2D.size()-1);
    // Draw a random point index with unsigned index = uniformDist(rng);
    
    cv::Matx34f bestP;
    unsigned bestInliers = 0;
    
    std::vector<cv::Vec3f> points2D_subset;
    points2D_subset.resize(subsetSize);
    std::vector<cv::Vec4f> points3D_subset;
    points3D_subset.resize(subsetSize);
    for (unsigned iter = 0; iter < numIterations; iter++) {
        for (unsigned j = 0; j < subsetSize; j++) {
            unsigned index = uniformDist(rng);
            points2D_subset[j] = points2D[index];
            points3D_subset[j] = points3D[index];
        }
        
        cv::Matx34f P = calibrate(points2D_subset, points3D_subset);

        unsigned numInliers = 0;
        for (unsigned i = 0; i < points2D.size(); i++) {
            cv::Vec3f projected = P * points3D[i];
            if (projected(2) > 0.0f) // in front
                if ((std::abs(points2D[i](0) - projected(0)/projected(2)) < threshold) &&
                    (std::abs(points2D[i](1) - projected(1)/projected(2)) < threshold))
                    numInliers++;
        }

        if (numInliers > bestInliers) {
            bestInliers = numInliers;
            bestP = P;
        }
    }
    
    return bestP;
}


// triangulates given set of image points based on projection matrices
/*
P1	projection matrix of first image
P2	projection matrix of second image
x1	image point set of first image
x2	image point set of second image
return	triangulated object points
*/
cv::Vec4f linearTriangulation(const cv::Matx34f& P1, const cv::Matx34f& P2, const cv::Vec3f& x1, const cv::Vec3f& x2)
{
    // allocate memory for design matrix
    Mat_<float> A(4, 4);

    // create design matrix
    // first row	x1(0, i) * P1(2, :) - P1(0, :)
    A(0, 0) = x1(0) * P1(2, 0) - P1(0, 0);
    A(0, 1) = x1(0) * P1(2, 1) - P1(0, 1);
    A(0, 2) = x1(0) * P1(2, 2) - P1(0, 2);
    A(0, 3) = x1(0) * P1(2, 3) - P1(0, 3);
    // second row	x1(1, i) * P1(2, :) - P1(1, :)
    A(1, 0) = x1(1) * P1(2, 0) - P1(1, 0);
    A(1, 1) = x1(1) * P1(2, 1) - P1(1, 1);
    A(1, 2) = x1(1) * P1(2, 2) - P1(1, 2);
    A(1, 3) = x1(1) * P1(2, 3) - P1(1, 3);
    // third row	x2(0, i) * P2(3, :) - P2(0, :)
    A(2, 0) = x2(0) * P2(2, 0) - P2(0, 0);
    A(2, 1) = x2(0) * P2(2, 1) - P2(0, 1);
    A(2, 2) = x2(0) * P2(2, 2) - P2(0, 2);
    A(2, 3) = x2(0) * P2(2, 3) - P2(0, 3);
    // first row	x2(1, i) * P2(3, :) - P2(1, :)
    A(3, 0) = x2(1) * P2(2, 0) - P2(1, 0);
    A(3, 1) = x2(1) * P2(2, 1) - P2(1, 1);
    A(3, 2) = x2(1) * P2(2, 2) - P2(1, 2);
    A(3, 3) = x2(1) * P2(2, 3) - P2(1, 3);

    cv::SVD svd(A);
    Mat_<float> tmp = svd.vt.row(3).t();

    return cv::Vec4f(tmp(0), tmp(1), tmp(2), tmp(3));
}

std::vector<cv::Vec4f> linearTriangulation(const cv::Matx34f& P1, const cv::Matx34f& P2, const std::vector<cv::Vec3f>& x1, const std::vector<cv::Vec3f>& x2)
{
    std::vector<cv::Vec4f> result;
    result.resize(x1.size());
    for (unsigned i = 0; i < result.size(); i++)
        result[i] = linearTriangulation(P1, P2, x1[i], x2[i]);
    return result;
}



void BundleAdjustment::BAState::computeResiduals(float *residuals) const
{
    unsigned rIdx = 0;
    for (unsigned camIdx = 0; camIdx < m_cameras.size(); camIdx++) {
        const auto &calibState = m_internalCalibs[m_scene.cameras[camIdx].internalCalibIdx];
        const auto &cameraState = m_cameras[camIdx];
        
        // TO DO !!!
        // Compute 3x4 camera matrix (composition of internal and external calibration)
        // Internal calibration is calibState.K
        // External calibration is dropLastRow(cameraState.H)
         
        cv::Mat P2_tmp = cv::Mat(calibState.K) * ((cv::Mat(cameraState.H))(Range(0, cameraState.H.rows - 1), Range(0, cameraState.H.cols)));
        //camera matrix
         cv::Matx34f P2 = (cv::Matx34f)P2_tmp;
        for (const KeyPoint &kp : m_scene.cameras[camIdx].keypoints) {
            const auto &trackState = m_tracks[kp.trackIdx];
            
            // Using P, compute the homogeneous position of the track in the image (world space position is trackState.location)
            cv::Vec3f projection = P2*trackState.location;
            
            // Compute the euclidean position of the track
            cv::Vec2f state_eucl(projection[0]/projection[2], projection[1]/ projection[2]);
            
            // Compute the residuals: the difference between computed position and real position (kp.location(0) and kp.location(1))
            // Compute and store the (signed!) residual in x direction multiplied by kp.weight
            residuals[rIdx++] = (kp.location(0) - state_eucl[0]) * kp.weight;

            // Compute and store the (signed!) residual in y direction multiplied by kp.weight
            residuals[rIdx++] = (kp.location(1) - state_eucl[1]) * kp.weight;
        }
    }
}

void BundleAdjustment::BAState::computeJacobiMatrix(JacobiMatrix *dst) const
{
    BAJacobiMatrix &J = dynamic_cast<BAJacobiMatrix&>(*dst);
    
    unsigned rIdx = 0;
    //multiple cameras
    for (unsigned camIdx = 0; camIdx < m_cameras.size(); camIdx++) {
        const auto &calibState = m_internalCalibs[m_scene.cameras[camIdx].internalCalibIdx];//Matx33
        const auto &cameraState = m_cameras[camIdx];  //Matx44f
        
        for (const KeyPoint &kp : m_scene.cameras[camIdx].keypoints) {
            const auto &trackState = m_tracks[kp.trackIdx];
            
            // calibState.K is the internal calbration
            // cameraState.H is the external calbration
            // trackState.location is the 3D location of the track in homogeneous coordinates

            // TO DO !!!
            // Compute the positions before and after the internal calibration (compare to slides).
 
            cv::Vec4f v_tmp = cameraState.H *trackState.location; 
            cv::Vec3f v(v_tmp[0],v_tmp[1],v_tmp[2]);// = ...Before calibration

            cv::Vec3f u = calibState.K * v;// After calibration.
            
            cv::Matx23f J_hom2eucl;
            // How do the euclidean image positions change when the homogeneous image positions change?
            J_hom2eucl(0,0) = 1.0f/u(2);
            J_hom2eucl(0,1) = 0.0f;
            J_hom2eucl(0,2) = - u(0)/(u(2)*u(2));
            J_hom2eucl(1,0) = 0.0f;
            J_hom2eucl(1,1) = 1.0f/u(2);
            J_hom2eucl(1,2) =  -u(1)/(u(2)*u(2));
            
            cv::Matx33f du_dDeltaK;
            
            // How do homogeneous image positions change when the internal calibration is changed (the 3 update parameters)?
            du_dDeltaK(0, 0) = v(0)*calibState.K(0,0);
            du_dDeltaK(0, 1) = v(2)* calibState.K(0,2);
            du_dDeltaK(0, 2) = 0.0;
            du_dDeltaK(1, 0) = v(1)* calibState.K(1,1);
            du_dDeltaK(1, 1) = 0.0;
            du_dDeltaK(1, 2) = v(2)* calibState.K(1,2);
            du_dDeltaK(2, 0) = 0.0;
            du_dDeltaK(2, 1) = 0.0;
            du_dDeltaK(2, 2) = 0.0;
            
            // Using the above (J_hom2eucl and du_dDeltaK), how do the euclidean image positions change when the internal calibration is changed (the 3 update parameters)?
            // Remember to include the weight of the keypoint (kp.weight)
            J.m_rows[rIdx].J_internalCalib = J_hom2eucl * du_dDeltaK * kp.weight ;
            
            // How do the euclidean image positions change when the tracks are moving in eye space/camera space (the vector "v" in the slides)?

            //cv::Mat K_34 = cv::Mat::zeros(3, 4, CV_32F);
            //K_34(cv::Range(0, 3), cv::Range(0, 3)) = (cv::Mat)calibState.K ;
            cv::Matx34f K_34;
            cv::Matx33f K = calibState.K;
            K_34(0,0)= K(0,0);
            K_34(0,1)= K(0,1);
            K_34(0,2)= K(0,2);
            K_34(0,3)= 0.f;

             K_34(1,0)= K(1,0);
             K_34(1,1)= K(1,1);
             K_34(1,2)= K(1,2);
             K_34(1,3)= 0.f;

             K_34(2,0)= K(2,0);
             K_34(2,1)= K(2,1);
             K_34(2,2)= K(2,2);
             K_34(2,3)= 0.f;

             cv::Matx<float, 2, 4> J_v2eucl =  J_hom2eucl * cv::Matx34f(K_34) ; // works like cv::Matx24f but the latter was not typedef-ed
            //Check
            
            //cv::Matx36f dv_dDeltaH;
            cv::Matx<float, 3, 6> dv_dDeltaH; // works like cv::Matx36f but the latter was not typedef-ed
            
            // How do tracks move in eye space (vector "v" in slides) when the parameters of the camera are changed?
            
            dv_dDeltaH(0, 0) = 0.f;
            dv_dDeltaH(0, 1) = v(2);
            dv_dDeltaH(0, 2) = -v(1);
            dv_dDeltaH(0, 3) = trackState.location(3);
            dv_dDeltaH(0, 4) = 0.f;
            dv_dDeltaH(0, 5) = 0.f;

            dv_dDeltaH(1, 0) = -v(2);
            dv_dDeltaH(1, 1) = 0.f;
            dv_dDeltaH(1, 2) = v(0);
            dv_dDeltaH(1, 3) = 0.f;
            dv_dDeltaH(1, 4) = trackState.location(3);
            dv_dDeltaH(1, 5) = 0.f;

            dv_dDeltaH(2, 0) = v(1);
            dv_dDeltaH(2, 1) = -v(0);
            dv_dDeltaH(2, 2) = 0.f;
            dv_dDeltaH(2, 3) = 0.f;
            dv_dDeltaH(2, 4) = 0.f;
            dv_dDeltaH(2, 5) = trackState.location(3);
        
            
            // How do the euclidean image positions change when the external calibration is changed (the 6 update parameters)?
            // Remember to include the weight of the keypoint (kp.weight)
            J.m_rows[rIdx].J_camera =  J_hom2eucl * calibState.K * dv_dDeltaH* kp.weight;//TODO: recheck!
             
             // How do the euclidean image positions change when the tracks are moving in world space (the x, y, z, and w before the external calibration)?
            // The multiplication operator "*" works as one would suspect. You can use dropLastRow(...) to drop the last row of a matrix.
            //cv::Mat H_34 = cv::Mat(cameraState.H)(cv::Range(0, 3), cv::Range(0, 4));
            cv::Matx<float, 2, 4> J_worldSpace2eucl = J_v2eucl * cameraState.H ;
            
             // How do the euclidean image positions change when the tracks are changed. 
            // This is the same as above, except it should also include the weight of the keypoint (kp.weight)
             J.m_rows[rIdx].J_track = J_worldSpace2eucl * kp.weight;
            
            rIdx++;
        }
    }
}

void BundleAdjustment::BAState::update(const float *update, State *dst) const
{               
    BAState &state = dynamic_cast<BAState &>(*dst);
    state.m_internalCalibs.resize(m_internalCalibs.size());
    state.m_cameras.resize(m_cameras.size());
    state.m_tracks.resize(m_tracks.size());
    
    unsigned intCalibOffset = 0;
    for (unsigned i = 0; i < m_internalCalibs.size(); i++) {
        state.m_internalCalibs[i].K = m_internalCalibs[i].K;

         /*
        * Modify the new internal calibration
        * 
        * m_internalCalibs[i].K is the old matrix, state.m_internalCalibs[i].K is the new matrix.
        * 
        * update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 0] is how much the focal length is supposed to change (scaled by the old focal length)
        * update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 1] is how much the principal point is supposed to shift in x direction (scaled by the old x position of the principal point)
        * update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 2] is how much the principal point is supposed to shift in y direction (scaled by the old y position of the principal point)
        */
		state.m_internalCalibs[i].K(0, 0) += update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 0] * m_internalCalibs[i].K(0, 0);
		state.m_internalCalibs[i].K(0, 2) += update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 1] * m_internalCalibs[i].K(0, 2);
		state.m_internalCalibs[i].K(1, 1) += update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 0] * m_internalCalibs[i].K(1, 1);
		state.m_internalCalibs[i].K(1, 2) += update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 2] * m_internalCalibs[i].K(1, 2);
		state.m_internalCalibs[i].K(2, 2) = 1;


    }
    unsigned cameraOffset = intCalibOffset + m_internalCalibs.size() * NumUpdateParams::INTERNAL_CALIB;
    for (unsigned i = 0; i < m_cameras.size(); i++) {
        // TO DO !!!
        /*
        * Compose the new matrix H
        * 
        * m_cameras[i].H is the old matrix, state.m_cameras[i].H is the new matrix.
        * 
        * update[cameraOffset + i * NumUpdateParams::CAMERA + 0] rotation increment around the camera X axis (not world X axis)
        * update[cameraOffset + i * NumUpdateParams::CAMERA + 1] rotation increment around the camera Y axis (not world Y axis)
        * update[cameraOffset + i * NumUpdateParams::CAMERA + 2] rotation increment around the camera Z axis (not world Z axis)
        * update[cameraOffset + i * NumUpdateParams::CAMERA + 3] translation increment along the camera X axis (not world X axis)
        * update[cameraOffset + i * NumUpdateParams::CAMERA + 4] translation increment along the camera Y axis (not world Y axis)
        * update[cameraOffset + i * NumUpdateParams::CAMERA + 5] translation increment along the camera Z axis (not world Z axis)
        * 
        * use rotationMatrixX(...), rotationMatrixY(...), rotationMatrixZ(...), and translationMatrix
        * 
        */
		state.m_cameras[i].H = rotationMatrixZ(update[cameraOffset + i * NumUpdateParams::CAMERA + 2])
				*rotationMatrixY(update[cameraOffset + i * NumUpdateParams::CAMERA + 1])
				*rotationMatrixX(update[cameraOffset + i * NumUpdateParams::CAMERA + 0])
				*translationMatrix(
				update[cameraOffset + i * NumUpdateParams::CAMERA + 3],
				update[cameraOffset + i * NumUpdateParams::CAMERA + 4],
				update[cameraOffset + i * NumUpdateParams::CAMERA + 5])*m_cameras[i].H;

    }
    unsigned trackOffset = cameraOffset + m_cameras.size() * NumUpdateParams::CAMERA;
    for (unsigned i = 0; i < m_tracks.size(); i++) {
        state.m_tracks[i].location = m_tracks[i].location;
        
        // TO DO !!!
        /*
        * Modify the new track location
        * 
        * m_tracks[i].location is the old location, state.m_tracks[i].location is the new location.
        * 
        * update[trackOffset + i * NumUpdateParams::TRACK + 0] increment of X
        * update[trackOffset + i * NumUpdateParams::TRACK + 1] increment of Y
        * update[trackOffset + i * NumUpdateParams::TRACK + 2] increment of Z
        * update[trackOffset + i * NumUpdateParams::TRACK + 3] increment of W
        */
       state.m_tracks[i].location(0) += update[trackOffset + i * NumUpdateParams::TRACK + 0] ;
       state.m_tracks[i].location(1) += update[trackOffset + i * NumUpdateParams::TRACK + 1] ;
       state.m_tracks[i].location(2) += update[trackOffset + i * NumUpdateParams::TRACK + 2] ;
       state.m_tracks[i].location(3) += update[trackOffset + i * NumUpdateParams::TRACK + 3] ;


        // Renormalization to length one
        float len = std::sqrt(state.m_tracks[i].location.dot(state.m_tracks[i].location));
        state.m_tracks[i].location *= 1.0f / len;
    }
}






/************************************************************************************************************/
/************************************************************************************************************/
/***************************                                     ********************************************/
/***************************    Nothing to do below this point   ********************************************/
/***************************                                     ********************************************/
/************************************************************************************************************/
/************************************************************************************************************/




BundleAdjustment::BAJacobiMatrix::BAJacobiMatrix(const Scene &scene)
{
    unsigned numResidualPairs = 0;
    for (const auto &camera : scene.cameras)
        numResidualPairs += camera.keypoints.size();
    
    m_rows.reserve(numResidualPairs);
    for (unsigned camIdx = 0; camIdx < scene.cameras.size(); camIdx++) {
        const auto &camera = scene.cameras[camIdx];
        for (unsigned kpIdx = 0; kpIdx < camera.keypoints.size(); kpIdx++) {
            m_rows.push_back({});
            m_rows.back().internalCalibIdx = camera.internalCalibIdx;
            m_rows.back().cameraIdx = camIdx;
            m_rows.back().keypointIdx = kpIdx;
            m_rows.back().trackIdx = camera.keypoints[kpIdx].trackIdx;
        }
    }
    
    m_internalCalibOffset = 0;
    m_cameraOffset = m_internalCalibOffset + scene.numInternalCalibs * NumUpdateParams::INTERNAL_CALIB;
    m_trackOffset = m_cameraOffset + scene.cameras.size() * NumUpdateParams::CAMERA;
    m_totalUpdateParams = m_trackOffset + scene.numTracks * NumUpdateParams::TRACK;
}

void BundleAdjustment::BAJacobiMatrix::multiply(float * __restrict dst, const float * __restrict src) const
{
    for (unsigned r = 0; r < m_rows.size(); r++) {
        float sumX = 0.0f;
        float sumY = 0.0f;
        for (unsigned i = 0; i < NumUpdateParams::INTERNAL_CALIB; i++) {
            sumX += src[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i] * 
                        m_rows[r].J_internalCalib(0, i);
            sumY += src[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i] * 
                        m_rows[r].J_internalCalib(1, i);
        }
        for (unsigned i = 0; i < NumUpdateParams::CAMERA; i++) {
            sumX += src[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i] * 
                        m_rows[r].J_camera(0, i);
            sumY += src[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i] * 
                        m_rows[r].J_camera(1, i);
        }
        for (unsigned i = 0; i < NumUpdateParams::TRACK; i++) {
            sumX += src[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i] * 
                        m_rows[r].J_track(0, i);
            sumY += src[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i] * 
                        m_rows[r].J_track(1, i);
        }
        dst[r*2+0] = sumX;
        dst[r*2+1] = sumY;
    }
}

void BundleAdjustment::BAJacobiMatrix::transposedMultiply(float * __restrict dst, const float * __restrict src) const
{
    memset(dst, 0, sizeof(float) * m_totalUpdateParams);
    // This is super ugly...
    for (unsigned r = 0; r < m_rows.size(); r++) {
        for (unsigned i = 0; i < NumUpdateParams::INTERNAL_CALIB; i++) {
            float elem = dst[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i];
            elem += src[r*2+0] * m_rows[r].J_internalCalib(0, i);
            elem += src[r*2+1] * m_rows[r].J_internalCalib(1, i);
            dst[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i] = elem;
        }
        
        for (unsigned i = 0; i < NumUpdateParams::CAMERA; i++) {
            float elem = dst[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i];
            elem += src[r*2+0] * m_rows[r].J_camera(0, i);
            elem += src[r*2+1] * m_rows[r].J_camera(1, i);
            dst[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i] = elem;
        }
        for (unsigned i = 0; i < NumUpdateParams::TRACK; i++) {
            float elem = dst[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i];
            elem += src[r*2+0] * m_rows[r].J_track(0, i);
            elem += src[r*2+1] * m_rows[r].J_track(1, i);
            dst[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i] = elem;
        }
    }
}

void BundleAdjustment::BAJacobiMatrix::computeDiagJtJ(float * __restrict dst) const
{
    memset(dst, 0, sizeof(float) * m_totalUpdateParams);
    // This is super ugly...
    for (unsigned r = 0; r < m_rows.size(); r++) {
        for (unsigned i = 0; i < NumUpdateParams::INTERNAL_CALIB; i++) {
            float elem = dst[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i];
            elem += m_rows[r].J_internalCalib(0, i) * m_rows[r].J_internalCalib(0, i);
            elem += m_rows[r].J_internalCalib(1, i) * m_rows[r].J_internalCalib(1, i);
            dst[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i] = elem;
        }
        for (unsigned i = 0; i < NumUpdateParams::CAMERA; i++) {
            float elem = dst[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i];
            elem += m_rows[r].J_camera(0, i) * m_rows[r].J_camera(0, i);
            elem += m_rows[r].J_camera(1, i) * m_rows[r].J_camera(1, i);
            dst[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i] = elem;
        }
        for (unsigned i = 0; i < NumUpdateParams::TRACK; i++) {
            float elem = dst[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i];
            elem += m_rows[r].J_track(0, i) * m_rows[r].J_track(0, i);
            elem += m_rows[r].J_track(1, i) * m_rows[r].J_track(1, i);
            dst[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i] = elem;
        }
    }
}



BundleAdjustment::BAState::BAState(const Scene &scene) : m_scene(scene)
{
    m_tracks.resize(m_scene.numTracks);
    m_internalCalibs.resize(m_scene.numInternalCalibs);
    m_cameras.resize(m_scene.cameras.size());
}

OptimizationProblem::State* BundleAdjustment::BAState::clone() const
{
    return new BAState(m_scene);
}


BundleAdjustment::BundleAdjustment(Scene &scene) : m_scene(scene)
{
    m_numResiduals = 0;
    for (const auto &camera : m_scene.cameras)
        m_numResiduals += camera.keypoints.size()*2;
    
    m_numUpdateParameters = 
                m_scene.numInternalCalibs * NumUpdateParams::INTERNAL_CALIB +
                m_scene.cameras.size() * NumUpdateParams::CAMERA +
                m_scene.numTracks * NumUpdateParams::TRACK;
}

OptimizationProblem::JacobiMatrix* BundleAdjustment::createJacobiMatrix() const
{
    return new BAJacobiMatrix(m_scene);
}


void BundleAdjustment::downweightOutlierKeypoints(BAState &state)
{
    std::vector<float> residuals;
    residuals.resize(m_numResiduals);
    state.computeResiduals(residuals.data());
    
    std::vector<float> distances;
    distances.resize(m_numResiduals/2);
    
    unsigned residualIdx = 0;
    for (auto &c : m_scene.cameras) {
        for (auto &kp : c.keypoints) {
            distances[residualIdx/2] = 
                std::sqrt(residuals[residualIdx+0]*residuals[residualIdx+0] + 
                          residuals[residualIdx+1]*residuals[residualIdx+1]);
            residualIdx+=2;
        }
    }

    std::vector<float> sortedDistances = distances;
    std::sort(sortedDistances.begin(), sortedDistances.end());
    
    std::cout << "min, max, median distances (weighted): " << sortedDistances.front() << " " << sortedDistances.back() << " " << sortedDistances[sortedDistances.size()/2] << std::endl;
    
    float thresh = sortedDistances[sortedDistances.size() * 2 / 3] * 2.0f;
    
    residualIdx = 0;
    for (auto &c : m_scene.cameras)
        for (auto &kp : c.keypoints) 
            if (distances[residualIdx++] > thresh) 
                kp.weight *= 0.5f;
}


Scene buildScene(const std::vector<std::string> &imagesFilenames)
{
    const float threshold = 20.0f;
    
    struct Image {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        
        std::vector<std::vector<std::pair<unsigned, unsigned>>> matches;
    };
    
    std::vector<Image> allImages;
    allImages.resize(imagesFilenames.size());
    Ptr<ORB> orb = ORB::create();
    orb->setMaxFeatures(10000);
    for (unsigned i = 0; i < imagesFilenames.size(); i++) {
        std::cout << "Extracting keypoints from " << imagesFilenames[i] << std::endl;
        cv::Mat img = cv::imread(imagesFilenames[i].c_str());
        orb->detectAndCompute(img, cv::noArray(), allImages[i].keypoints, allImages[i].descriptors);
        allImages[i].matches.resize(allImages[i].keypoints.size());
    }
    
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);
    for (unsigned i = 0; i < allImages.size(); i++)
        for (unsigned j = i+1; j < allImages.size(); j++) {
            std::cout << "Matching " << imagesFilenames[i] << " against " << imagesFilenames[j] << std::endl;
            
            std::vector<std::vector<cv::DMatch>> matches;
            matcher->knnMatch(allImages[i].descriptors, allImages[j].descriptors, matches, 2);
            for (unsigned k = 0; k < matches.size(); ) {
                if (matches[k][0].distance > matches[k][1].distance * 0.75f) {
                    matches[k] = std::move(matches.back());
                    matches.pop_back();
                } else k++;
            }
            std::vector<cv::Vec3f> p1, p2;
            p1.resize(matches.size());
            p2.resize(matches.size());
            for (unsigned k = 0; k < matches.size(); k++) {
                p1[k] = cv::Vec3f(allImages[i].keypoints[matches[k][0].queryIdx].pt.x,
                                  allImages[i].keypoints[matches[k][0].queryIdx].pt.y,
                                  1.0f);
                p2[k] = cv::Vec3f(allImages[j].keypoints[matches[k][0].trainIdx].pt.x,
                                  allImages[j].keypoints[matches[k][0].trainIdx].pt.y,
                                  1.0f);
            }
            std::cout << "RANSACing " << imagesFilenames[i] << " against " << imagesFilenames[j] << std::endl;
            
            cv::Matx33f F = estimateFundamentalRANSAC(p1, p2, 1000, threshold);
            
            std::vector<std::pair<unsigned, unsigned>> inlierMatches;
            for (unsigned k = 0; k < matches.size(); k++) 
                if (getError(p1[k], p2[k], F) < threshold) 
                    inlierMatches.push_back({
                        matches[k][0].queryIdx,
                        matches[k][0].trainIdx
                    });
            const unsigned minMatches = 400;
                
            std::cout << "Found " << inlierMatches.size() << " valid matches!" << std::endl;
            if (inlierMatches.size() >= minMatches)
                for (const auto p : inlierMatches) {
                    allImages[i].matches[p.first].push_back({j, p.second});
                    allImages[j].matches[p.second].push_back({i, p.first});
                }
        }
    
    
    Scene scene;
    scene.numInternalCalibs = 1;
    scene.cameras.resize(imagesFilenames.size());
    for (auto &c : scene.cameras)
        c.internalCalibIdx = 0;
    scene.numTracks = 0;
    
    std::cout << "Finding tracks " << std::endl;
    {
        std::set<std::pair<unsigned, unsigned>> handledKeypoints;
        std::set<unsigned> imagesSpanned;
        std::vector<std::pair<unsigned, unsigned>> kpStack;
        std::vector<std::pair<unsigned, unsigned>> kpList;
        for (unsigned i = 0; i < allImages.size(); i++) {
            for (unsigned kp = 0; kp < allImages[i].keypoints.size(); kp++) {
                if (allImages[i].matches[kp].empty()) continue;
                if (handledKeypoints.find({i, kp}) != handledKeypoints.end()) continue;
                
                bool valid = true;
                
                kpStack.push_back({i, kp});
                while (!kpStack.empty()) {
                    auto kp = kpStack.back();
                    kpStack.pop_back();
                    
                    
                    if (imagesSpanned.find(kp.first) != imagesSpanned.end()) // appearing twice in one image -> invalid
                        valid = false;
                    
                    handledKeypoints.insert(kp);
                    kpList.push_back(kp);
                    imagesSpanned.insert(kp.first);
                    
                    for (const auto &matchedKp : allImages[kp.first].matches[kp.second])
                        if (handledKeypoints.find(matchedKp) == handledKeypoints.end()) 
                            kpStack.push_back(matchedKp);
                }
                
                if (valid) {
                    //std::cout << "Forming track from group of " << kpList.size() << " keypoints over " << imagesSpanned.size() << " images" << std::endl;
                    
                    for (const auto &kp : kpList) {
                        cv::Vec2f pixelPosition;
                        pixelPosition(0) = allImages[kp.first].keypoints[kp.second].pt.x;
                        pixelPosition(1) = allImages[kp.first].keypoints[kp.second].pt.y;
                        
                        unsigned trackIdx = scene.numTracks;
                        
                        scene.cameras[kp.first].keypoints.push_back({
                            pixelPosition,
                            trackIdx,
                            1.0f
                        });
                    }
                    
                    scene.numTracks++;
                } else {
                    //std::cout << "Dropping invalid group of " << kpList.size() << " keypoints over " << imagesSpanned.size() << " images" << std::endl;
                }
                kpList.clear();
                imagesSpanned.clear();
            }
        }
        std::cout << "Formed " << scene.numTracks << " tracks" << std::endl;
    }
    
    for (auto &c : scene.cameras)
        if (c.keypoints.size() < 100)
            std::cout << "Warning: One camera is connected with only " << c.keypoints.size() << " keypoints, this might be too unstable!" << std::endl;

    return scene;
}

void produceInitialState(const Scene &scene, const cv::Matx33f &initialInternalCalib, BundleAdjustment::BAState &state)
{
    const float threshold = 20.0f;
    
    state.m_internalCalibs[0].K = initialInternalCalib;
    
    std::set<unsigned> triangulatedPoints;
    
    const unsigned image1 = 0;
    const unsigned image2 = 1;
    // Find stereo pose of first two images
    {
        
        std::map<unsigned, cv::Vec2f> track2keypoint;
        for (const auto &kp : scene.cameras[image1].keypoints)
            track2keypoint[kp.trackIdx] = kp.location;
        
        std::vector<std::pair<cv::Vec2f, cv::Vec2f>> matches;
        std::vector<unsigned> matches2track;
        for (const auto &kp : scene.cameras[image2].keypoints) {
            auto it = track2keypoint.find(kp.trackIdx);
            if (it != track2keypoint.end()) {
                matches.push_back({it->second, kp.location});
                matches2track.push_back(kp.trackIdx);
            }
        }
        
        std::cout << "Initial pair has " << matches.size() << " matches" << std::endl;
        
        std::vector<cv::Vec3f> p1;
        p1.reserve(matches.size());
        std::vector<cv::Vec3f> p2;
        p2.reserve(matches.size());
        for (unsigned i = 0; i < matches.size(); i++) {
            p1.push_back(cv::Vec3f(matches[i].first(0), matches[i].first(1), 1.0f));
            p2.push_back(cv::Vec3f(matches[i].second(0), matches[i].second(1), 1.0f));
        }
        
        const cv::Matx33f &K = initialInternalCalib;
        state.m_cameras[image1].H = cv::Matx44f::eye();
        state.m_cameras[image2].H = computeCameraPose(K, p1, p2);
            
        std::vector<cv::Vec4f> X = linearTriangulation(K * cv::Matx34f::eye(), K * cv::Matx34f::eye() * state.m_cameras[image2].H, p1, p2);
        for (unsigned i = 0; i < X.size(); i++) {
            cv::Vec4f t = X[i];
            t /= std::sqrt(t.dot(t));
            state.m_tracks[matches2track[i]].location = t;
            triangulatedPoints.insert(matches2track[i]);
        }
    }
    

    for (unsigned c = 0; c < scene.cameras.size(); c++) {
        if (c == image1) continue;
        if (c == image2) continue;
        
        std::vector<KeyPoint> triangulatedKeypoints;
        for (const auto &kp : scene.cameras[c].keypoints) 
            if (triangulatedPoints.find(kp.trackIdx) != triangulatedPoints.end()) 
                triangulatedKeypoints.push_back(kp);

        if (triangulatedKeypoints.size() < 100)
            std::cout << "Warning: Camera " << c << " is only estimated from " << triangulatedKeypoints.size() << " keypoints" << std::endl;
        
        std::vector<cv::Vec3f> points2D;
        points2D.resize(triangulatedKeypoints.size());
        std::vector<cv::Vec4f> points3D;
        points3D.resize(triangulatedKeypoints.size());
        
        for (unsigned i = 0; i < triangulatedKeypoints.size(); i++) {
            points2D[i] = cv::Vec3f(
                        triangulatedKeypoints[i].location(0),
                        triangulatedKeypoints[i].location(1),
                        1.0f);
            points3D[i] = state.m_tracks[triangulatedKeypoints[i].trackIdx].location;
        }
        
        std::cout << "Estimating camera " << c << " from " << triangulatedKeypoints.size() << " keypoints" << std::endl;
        //cv::Mat P = calibrate(points2D, points3D);
        cv::Matx34f P = estimateProjectionRANSAC(points2D, points3D, 1000, threshold);
        cv::Matx33f K, R;
        ProjectionMatrixInterpretation info;
        interprete(P, K, R, info);
        
        state.m_cameras[c].H = cv::Matx44f::eye();
        for (unsigned i = 0; i < 3; i++)
            for (unsigned j = 0; j < 3; j++)
                state.m_cameras[c].H(i, j) = R(i, j);
            
        state.m_cameras[c].H = state.m_cameras[c].H * translationMatrix(-info.cameraLocation[0], -info.cameraLocation[1], -info.cameraLocation[2]);
    }
    // Triangulate remaining points
    for (unsigned c = 0; c < scene.cameras.size(); c++) {
        
        cv::Matx34f P1 = state.m_internalCalibs[scene.cameras[c].internalCalibIdx].K * cv::Matx34f::eye() * state.m_cameras[c].H;
            
        for (unsigned otherC = 0; otherC < c; otherC++) {
            cv::Matx34f P2 = state.m_internalCalibs[scene.cameras[otherC].internalCalibIdx].K * cv::Matx34f::eye() * state.m_cameras[otherC].H;
            for (const auto &kp : scene.cameras[c].keypoints) {
                if (triangulatedPoints.find(kp.trackIdx) != triangulatedPoints.end()) continue;
                
                for (const auto &otherKp : scene.cameras[otherC].keypoints) {
                    if (kp.trackIdx == otherKp.trackIdx) {
                        cv::Vec4f X = linearTriangulation(
                            P1, P2,
                            cv::Vec3f(kp.location(0), kp.location(1), 1.0f),
                            cv::Vec3f(otherKp.location(0), otherKp.location(1), 1.0f)
                        );
                        
                        X /= std::sqrt(X.dot(X));
                        state.m_tracks[kp.trackIdx].location = X;
                        
                        triangulatedPoints.insert(kp.trackIdx);
                    }
                }
            }
        }
    }
    if (triangulatedPoints.size() != state.m_tracks.size())
        std::cout << "Warning: Some tracks were not triangulated. This should not happen!" << std::endl;
}


}
