#include "Helper.h"

#include <vector>
#include <string>

#include <fstream>

namespace pcv4 {

struct WinInfo { 
    cv::Mat img; 
    std::string name; 
    std::vector<cv::Vec3f> pointList; 
};



// mouse call back to get points and draw circles
/*
event	specifies encountered mouse event
x,y	position of mouse pointer
flags	not used here
param	a struct containing used IplImage and window title
*/
void getPointsCB(int event, int x, int y, int flags, void* param){

    // cast to a structure
    WinInfo* win = (WinInfo*) param;

    switch(event){
        // if left mouse button was pressed
        case cv::EVENT_LBUTTONDOWN:{
            // create point representing mouse position
            cv::Point2f p(x,y);
            // draw green point
            cv::circle(win->img, p, 2, cv::Scalar(0, 255, 0), 2);
            // draw green circle
            cv::circle(win->img, p, 15, cv::Scalar(0, 255, 0), 2);
            // update image
            cv::imshow(win->name.c_str(), win->img);
            // add point to point list
            win->pointList.push_back({(float) x, (float) y, 1.0f});
        }break;
    }
}


int getPointsManual(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Vec3f>& p1, std::vector<cv::Vec3f>& p2)
{
    std::cout << std::endl;
    std::cout << "Please select at least four points by clicking at the corresponding image positions:" << std::endl;
    std::cout << "Firstly click at the point that shall be transformed (within the image to be attached), followed by a click on the corresponding point within the base image" << std::endl;
    std::cout << "Continue until you have collected as many point pairs as you wish" << std::endl;
    std::cout << "Stop the point selection by pressing any key" << std::endl << std::endl;

    
    WinInfo windowInfoBase = {
        img1.clone(),
        "Image 1",
        {}
    };
    
    WinInfo windowInfoAttach = {
        img2.clone(),
        "Image 2",
        {}
    };
    
    
    // show input images and install mouse callback
    cv::namedWindow( windowInfoBase.name.c_str(), 0 );
    cv::imshow( windowInfoBase.name.c_str(), windowInfoBase.img );
    cv::setMouseCallback(windowInfoBase.name.c_str(), getPointsCB, (void*) &windowInfoBase);
    

    cv::namedWindow( windowInfoAttach.name.c_str(), 0 );
    cv::imshow( windowInfoAttach.name.c_str(), windowInfoAttach.img );
    cv::setMouseCallback(windowInfoAttach.name.c_str(), getPointsCB, (void*) &windowInfoAttach);

    // wait until any key was pressed
    cv::waitKey(0);
    
    cv::destroyWindow( windowInfoBase.name.c_str() );
    cv::destroyWindow( windowInfoAttach.name.c_str() );

    int numOfPoints = windowInfoBase.pointList.size();
    p1 = std::move(windowInfoBase.pointList);
    p2 = std::move(windowInfoAttach.pointList);
    return numOfPoints;
}


cv::Mat drawMatches(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Vec3f>& p1, std::vector<cv::Vec3f>& p2)
{
    
    cv::Mat_<cv::Vec3b> img1Converted;
    img1.convertTo(img1Converted, CV_8UC3);
    cv::Mat_<cv::Vec3b> img2Converted;
    img2.convertTo(img2Converted, CV_8UC3);
    
    cv::Mat_<cv::Vec3b> image(img1.rows + img2.rows, std::max(img1.cols, img2.cols));
    img1Converted.copyTo(image(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2Converted.copyTo(image(cv::Rect(0, img1.rows, img2.cols, img2.rows)));


    if (p1.size() != p2.size())
        throw std::runtime_error("Mismatched sizes for matched point arrays!");
    
    for (unsigned i = 0; i < p1.size(); i++) {
        cv::line(image, {p1[i][0]/p1[i][2], p1[i][1]/p1[i][2]}, {p2[i][0]/p2[i][2], img1.rows+p2[i][1]/p2[i][2]}, cv::Scalar(0,0,255), 1);
    }
    
    // draw circles on top
    for (unsigned i = 0; i < p1.size(); i++) {
        cv::circle(image, {p1[i][0]/p1[i][2], p1[i][1]/p1[i][2]}, 2, cv::Scalar(0, 255, 0), 2);
        cv::circle(image, {p2[i][0]/p2[i][2], img1.rows+p2[i][1]/p2[i][2]}, 2, cv::Scalar(0, 255, 0), 2);
    }
    return image;
}


/** 
 * @brief Draws line given in homogeneous representation into image
 * @param img the image to draw into
 * @param a The line parameters
 * @param b The line parameters
 * @param c The line parameters
 */
void drawEpiLine(cv::Mat& img, double a, double b, double c)
{

    // calculate intersection with image borders
    cv::Point p1 = cv::Point(-c/a, 0);						// schnittpunkt mit unterer bildkante (x-achse)
    cv::Point p2 = cv::Point(0, -c/b);						// schnittpunkt mit linker bildkante (y-achse)
    cv::Point p3 = cv::Point((-b*(img.rows-1)-c)/a, img.rows-1);		// schnittpunkt mit oberer bildkante
    cv::Point p4 = cv::Point(img.cols-1, (-a*(img.cols-1)-c)/b);		// schnittpunkt mit rechter bildkante

    // check start and end points
    cv::Point startPoint, endPoint, cur_p;
    startPoint.x = startPoint.y = endPoint.x = endPoint.y = 0;
    bool set_start = false;
    for(int p=0; p<4; p++){
        switch(p){
            case 0: cur_p = p1; break;
            case 1: cur_p = p2; break;
            case 2: cur_p = p3; break;
            case 3: cur_p = p4; break;
        }
        if ( (cur_p.x >= 0) and (cur_p.x < img.cols) and (cur_p.y >= 0) and (cur_p.y < img.rows) ){
            if (!set_start){
                startPoint = cur_p;
                set_start = true;
            }else{
                endPoint = cur_p;
            }
        }
    }

    // draw line
    cv::line(img, startPoint, endPoint, cv::Scalar(0,0,255), 1);
}


RawOrbMatches extractRawOrbMatches(const cv::Mat& img1, const cv::Mat& img2)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->setMaxFeatures(30000);
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
    
    RawOrbMatches result;
    result.keypoints1.reserve(keypoints1.size());
    for (const auto &kp : keypoints1)
        result.keypoints1.push_back({kp.pt.x, kp.pt.y, 1.0f});

    result.keypoints2.reserve(keypoints2.size());
    for (const auto &kp : keypoints2)
        result.keypoints2.push_back({kp.pt.x, kp.pt.y, 1.0f});
    
    
    std::vector<std::vector<cv::DMatch>> matches;
    
    matcher->knnMatch(descriptors1, descriptors2, matches, 2);
    for (const auto &m : matches) {
        RawOrbMatches::Match match;
        match.closest = m[0].trainIdx;
        match.closestDistance = m[0].distance;
        match.secondClosest = m[1].trainIdx;
        match.secondClosestDistance = m[1].distance;
        result.matches_1_2[m[0].queryIdx] = match;
    }
    
    matcher->knnMatch(descriptors2, descriptors1, matches, 2);
    for (const auto &m : matches) {
        RawOrbMatches::Match match;
        match.closest = m[0].trainIdx;
        match.closestDistance = m[0].distance;
        match.secondClosest = m[1].trainIdx;
        match.secondClosestDistance = m[1].distance;
        result.matches_2_1[m[0].queryIdx] = match;
    }

    return result;
}



}
