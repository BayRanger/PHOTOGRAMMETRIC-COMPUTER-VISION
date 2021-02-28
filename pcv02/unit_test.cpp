//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Pcv2.h"

#include <opencv2/opencv.hpp>

#include <iostream>


using namespace std;
using namespace pcv2;
using namespace cv;



void test_getCondition2D(void){

    std::vector<cv::Vec3f> p = {
        {93.0f, 617.0f, 1.0f},
        {729.0f, 742.0f, 1.0f},
        {703.0f, 1233.0f, 1.0f}, 
        {152.0f, 1103.0f, 1.0f},
    };
    Matx33f Ttrue(
            1./296.75, 0, -419.25/296.75, 
            0, 1./244.25, -923.75/244.25, 
            0, 0, 1);
    
    Matx33f Test = getCondition2D(p);
    if (std::abs(Test(2,2)) < 1e-4f) {
        cout << "Warning: There seems to be a problem with getCondition2D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cin.get();
        exit(-1);
    }
    Test *= 1.0f / Test(2,2);
    const float eps = 1e-3f;
    if (sum(abs(Mat(Test - Ttrue))).val[0] > eps){
        cout << "Warning: There seems to be a problem with getCondition2D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cin.get();
        exit(-1);
    }
}

void test_getDesignMatrix_homography2D(void)
{
    std::vector<cv::Vec3f> p1 = {
        {-1.0f, -1.0f, 1.0f},
        {1.0f, -1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f},
    };

    std::vector<cv::Vec3f> p2 = {
        {-1.0994103, -1.2558856, 1.0f},
        {1.0438079, -0.74411488, 1.0f}, 
        {0.9561919, 1.2661204, 1.0f},
        {-0.90058976, 0.73387909, 1.0f},
    };
    
    Mat_<float> Aest = getDesignMatrix_homography2D(p1, p2);
    if ( ( (Aest.rows != 8) && (Aest.rows != 9) ) || (Aest.cols != 9)){
        cout << "Warning: There seems to be a problem with getDesignMatrix_homography2D(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cin.get();
        exit(-1);
    }
    Mat Atrue;
    if (Aest.rows == 8)
        Atrue = (Mat_<float>(8,9) << 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 1.0994103, 1.2558856, -1, -1.0438079, 0.74411488, -1, 0, 0, 0, 1.0438079, -0.74411488, 1, 0, 0, 0, -1.0438079, 0.74411488, -1, -1.0438079, 0.74411488, -1, -0.9561919, -1.2661204, -1, 0, 0, 0, 0.9561919, 1.2661204, 1, 0, 0, 0, -0.9561919, -1.2661204, -1, 0.9561919, 1.2661204, 1, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, -0.90058976, 0.73387909, 1);
    else
        Atrue = (Mat_<float>(9,9) << 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 1.0994103, 1.2558856, -1, -1.0438079, 0.74411488, -1, 0, 0, 0, 1.0438079, -0.74411488, 1, 0, 0, 0, -1.0438079, 0.74411488, -1, -1.0438079, 0.74411488, -1, -0.9561919, -1.2661204, -1, 0, 0, 0, 0.9561919, 1.2661204, 1, 0, 0, 0, -0.9561919, -1.2661204, -1, 0.9561919, 1.2661204, 1, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, -0.90058976, 0.73387909, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const float eps = 1e-3f;
    if (sum(abs(Aest - Atrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with getDesignMatrix_homography2D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cin.get();
        exit(-1);
    }
}

void test_solve_dlt(void){
    Mat_<float> A(8,9);
    A << 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 1.0994103, 1.2558856, -1, -1.0438079, 0.74411488, -1, 0, 0, 0, 1.0438079, -0.74411488, 1, 0, 0, 0, -1.0438079, 0.74411488, -1, -1.0438079, 0.74411488, -1, -0.9561919, -1.2661204, -1, 0, 0, 0, 0.9561919, 1.2661204, 1, 0, 0, 0, -0.9561919, -1.2661204, -1, 0.9561919, 1.2661204, 1, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, -0.90058976, 0.73387909, 1;
    
    cv::Matx33f Hest = solve_dlt_homography2D(A);
    Hest *= 1.0f / Hest(2,2);
    Matx33f Htrue(0.57111752, -0.017852778, 0.013727478, 
                -0.15091757, 0.57065326, -0.04098846, 
                0.024604173, -0.041672569, 0.56645769);
    Htrue *= 1.0f / Htrue(2,2);
    const float eps = 1e-3f;
    
    if (sum(abs(Mat(Hest - Htrue))).val[0] > eps){
        cout << "Warning: There seems to be a problem with solve_dlt_homography2D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cin.get();
        exit(-1);
    }
}

void test_decondition(void){
    
    Matx33f H(0.57111752, -0.017852778, 0.013727478, 
             -0.15091757, 0.57065326, -0.04098846, 
              0.024604173, -0.041672569, 0.56645769);
    Matx33f T1(1./319.5, 0, -1, 0, 1./319.5, -1, 0, 0, 1);
    Matx33f T2(1./296.75, 0, -419.25/296.75, 0, 1./244.25, -923.75/244.25, 0, 0, 1);
    H = decondition_homography2D(T1, T2, H);
    H *= 1.0f / H(2,2);
    Matx33f Htrue(0.9304952, -0.11296108, -16.839279, -0.19729686, 1.003845, -601.02362, 0.00012028422, -0.00024751772, 1);
    const float eps = 1e-3f;
    if (sum(abs(Mat(H - Htrue))).val[0] > eps){
        cout << "Warning: There seems to be a problem with decondition_homography2D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cin.get();
        exit(-1);
    }
}

void test_homography2D(void){


    std::vector<cv::Vec3f> p1 = {
        {0.0f, 0.0f, 1.0f},
        {639.0f, 0.0f, 1.0f},
        {639.0f, 639.0f, 1.0f}, 
        {0.0f, 639.0f, 1.0f},
    };

    std::vector<cv::Vec3f> p2 = {
        {93.0f, 617.0f, 1.0f},
        {729.0f, 742.0f, 1.0f},
        {703.0f, 1233.0f, 1.0f}, 
        {152.0f, 1103.0f, 1.0f},
    };
    
    Matx33f Hest = homography2D(p1, p2);
    if (std::abs(Hest(2,2)) < 1e-4f) {
        cout << "Warning: There seems to be a problem with homography2D(..)!" << endl;
        cout << "\t==> Expected H(2,2) to be non-zero!" << endl;
        cin.get();
        exit(-1);
    }
    Hest *= 1.0f / Hest(2,2);
    Matx33f Htrue(0.9304952, -0.11296108, -16.839279, -0.19729686, 1.003845, -601.02362, 0.00012028422, -0.00024751772, 1);
    const float eps = 1e-3f;
    if (sum(abs(Mat(Hest - Htrue))).val[0] > eps){
        cout << "Warning: There seems to be a problem with homography2D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cin.get();
        exit(-1);
    }
}



int main(int argc, char** argv) {

    test_getCondition2D();
    test_getDesignMatrix_homography2D();
    test_solve_dlt();
    test_decondition();
    test_homography2D();

    cout << "Finished basic testing: Everything seems to be fine." << endl;
    cin.get();
    return 0;

}
