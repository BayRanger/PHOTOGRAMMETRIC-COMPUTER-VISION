//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Pcv3.h"

#include "Helper.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>


using namespace std;

// function loads input image, calls processing function, and saves result
/*
fname	path to input image
*/
void run(std::string imgPath, std::string pointPath){

    // load image calibration image, paths in argv[1]
    cv::Mat calibImage = cv::imread(imgPath);
    if (!calibImage.data){
        cerr << "ERROR: Could not load image " << imgPath << endl;
        cerr << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }

    // get corresponding points within the image
    std::vector<cv::Vec3f> points2D;
    std::vector<cv::Vec4f> points3D;
    std::string fname = pointPath;
    int numberOfPointPairs = pcv3::getPoints(calibImage, fname, points2D, points3D);

    // just some putput
    cout << "Number of defined point pairs: " << numberOfPointPairs << endl;
    cout << endl << "2D Points in image:" << endl;
    for (const auto &p : points2D)
        cout << p << endl;
    cout << endl << "3D Points at object:" << endl;
    for (const auto &p : points3D)
        cout << p << endl;
    
    // calculate projection matrix
    cv::Matx34f P = pcv3::calibrate(points2D, points3D);
    
    // decompose P to get camera parameter
    cv::Matx33f K;
    cv::Matx33f R;
    pcv3::ProjectionMatrixInterpretation info;
    pcv3::interprete(P, K, R, info);
    
    cout << endl << "Calibration matrix: " << endl;
    cout << K << endl;
    cout << endl << "Rotation matrix: " << endl;
    cout << R << endl;

    cout << endl << info;
}


// usage: path to image in argv[1]
// main function. loads and saves image
int main(int argc, char** argv) {

    // will contain path to input image (taken from argv[1])
    std::string imgPath, pointPath;

    // check if image paths are defined
    if (argc != 3){
        cerr << "Usage: ./main <path to calibration image> <path to object-point file>" << endl;
        cerr << "Press enter to continue..." << endl;
        cin.get();
        return -1;
    }else{
        // if yes, assign it to variable fname
        imgPath = argv[1];
        pointPath = argv[2];
    }

    // start processing
    run(imgPath, pointPath);

    cout << "Press enter to continue..." << endl;
    cin.get();

    return 0;

}
