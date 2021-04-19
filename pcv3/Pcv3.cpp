//============================================================================
// Name        : Pcv3.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : Camera calibration
//============================================================================

#include "Pcv3.h"

namespace pcv3 {

/**
 * @brief get the conditioning matrix of given points
 * @param the points as matrix
 * @returns the condition matrix (already allocated)
 */
cv::Matx33f getCondition2D(const std::vector<cv::Vec3f> &points)
{
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
 * @brief get the conditioning matrix of given points
 * @param the points as matrix
 * @returns the condition matrix (already allocated)
 */
cv::Matx44f getCondition3D(const std::vector<cv::Vec4f> &points)
{
    // TO DO !!!
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

 switch (type) {
        case GEOM_TYPE_POINT: {
            // TO DO !!!
            for(int i = 0; i < geomObjects.size(); i++){

                cv::Vec3f geopoint = geomObjects[i];
                cv::Vec3f temp;


                temp[0] = geopoint[0] * H(0,0) + geopoint[1] * H(0,1) + H(0,2);
                temp[1] = geopoint[0] * H(1,0) + geopoint[1] * H(1,1) + H(1,2);
                temp[2] = 1;

                result.push_back(temp);

            }
        } break;
        case GEOM_TYPE_LINE: {
            // TO DO !!!
            cv::Matx33f transposed, inverse;

            transposed = H.t();
            inverse = transposed.inv();

            for(int i = 0; i < geomObjects.size(); i++) {

                cv::Vec3f resLine, tempPoint;

                tempPoint = geomObjects[i];

                resLine[0] = tempPoint[0] * inverse(0,0) + tempPoint[1] * inverse(0,1) + tempPoint[2] * inverse(0,2);
                resLine[1] = tempPoint[0] * inverse(1,0) + tempPoint[1] * inverse(1,1) + tempPoint[2] * inverse(1,2);
                resLine[2] = tempPoint[0] * inverse(2,0) + tempPoint[1] * inverse(2,1) + tempPoint[2] * inverse(2,2);

                result.push_back(resLine);

            }
        } break;
        default:
            throw std::runtime_error("Unhandled geometry type!");
    }
    return result;
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
 * @brief Define the design matrix as needed to compute projection matrix
 * @param points2D Set of 2D points within the image
 * @param points3D Set of 3D points at the object
 * @returns The design matrix to be computed
 */
cv::Mat_<float> getDesignMatrix_camera(const std::vector<cv::Vec3f>& points2D, const std::vector<cv::Vec4f>& points3D)
{
    // TO DO !!!
      // TO DO !!!
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
    //return cv::Mat_<float>(2*points2D.size(), 12);
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
 * @param P The 3x4 projection matrix, only "input" to this function
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
    //triangular matrix R and an orthogonal matrix Q

    //float omega_degree, phi_degree, kappa_degree;

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




}
