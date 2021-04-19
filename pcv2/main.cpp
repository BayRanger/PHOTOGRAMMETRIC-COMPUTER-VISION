//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Pcv2.h"
#include "Helper.h"

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;


// function loads input image, calls processing function, and saves result
/*
fname	path to input image
*/
void run(const std::string &fnameBase, const std::string &fnameLeft, const std::string &fnameRight) {

    // load image first two images, paths in argv[1] and argv[2]
    cv::Mat baseImage = cv::imread(fnameBase);
    cv::Mat attachImage = cv::imread(fnameLeft);
    if (!baseImage.data){
        cerr << "ERROR: Cannot read image ( " << fnameBase << endl;
        cin.get();
        exit(-1);
    }
    if (!attachImage.data){
        cerr << "ERROR: Cannot read image ( " << fnameLeft << endl;
        cin.get();
        exit(-1);
    }

    // get corresponding points within the two image
    // start with one point within the attached image, then click on corresponding point in base image
    std::vector<cv::Vec3f> p_basis, p_attach;
    int numberOfPointPairs = pcv2::getPoints(baseImage, attachImage, p_basis, p_attach);
    
    // just some putput
    cout << "Number of defined point pairs: " << numberOfPointPairs << endl;
    cout << endl << "Points in base image:" << endl;
    for (const auto &p : p_basis)
        cout << p << endl;
    cout << endl << "Points in second image:" << endl;
    for (const auto &p : p_attach)
        cout << p << endl;

    // calculate homography
    cv::Matx33f H = pcv2::homography2D(p_basis, p_attach);
    
    // create panorama
    cv::Mat panorama = pcv2::stitch(baseImage, attachImage, H);

    const char *windowName = "Panorama";
    
    // display panorama (resizeable)
    cv::namedWindow( windowName, 0 );
    cv::imshow(windowName, panorama );
    cv::waitKey(0);
    cv::destroyWindow(windowName);
    
    // panorama is new base image, third image is the image to attach
    baseImage = panorama;
    // load third image
    attachImage = cv::imread(fnameRight);
    if (!attachImage.data){
        cout << "ERROR: Cannot read image ( " << fnameRight << " )" << endl;
        cin.get();
        exit(-1);
    }
    
    // get corresponding points within the two image
    // start with one point within the attached image, then click on corresponding point in base image
    numberOfPointPairs = pcv2::getPoints(baseImage, attachImage, p_basis, p_attach);
    
    // just some putput
    cout << "Number of defined point pairs: " << numberOfPointPairs << endl;
    cout << endl << "Points in base image:" << endl;
    for (const auto &p : p_basis)
        cout << p << endl;
    cout << endl << "Points in second image:" << endl;
    for (const auto &p : p_attach)
        cout << p << endl;

    
    // calculate homography
    H = pcv2::homography2D(p_basis, p_attach);
    
    // create panorama
    panorama = pcv2::stitch(baseImage, attachImage, H);
    
    // display panorama (resizeable)
    cv::namedWindow( windowName, 0 );
    cv::imshow(windowName, panorama );
    cv::waitKey(0);
    cv::destroyWindow(windowName);
    
    cv::imwrite("panorama.png", panorama);
}





// usage: path to image in argv[1]
// main function. loads and saves image
int main(int argc, char** argv) {

    // will contain path to the input image (taken from argv[1])
    string fnameBase, fnameLeft, fnameRight;

    // check if image paths are defined
    if (argc != 4){
        cout << "Usage: pcv2 <path to base image> <path to 2nd image> <path to 3rd image>" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        return -1;
    }else{
        // if yes, assign it to variable fname
        fnameBase = argv[1];
        fnameLeft = argv[2];
        fnameRight = argv[3];
    }
    
    // start processing
    run(fnameBase, fnameLeft, fnameRight);

    cout << "Press enter to continue..." << endl;
    cin.get();

    return 0;

}
