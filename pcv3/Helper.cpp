#include "Helper.h"

#include <vector>
#include <string>

#include <fstream>

namespace pcv3 {

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
            win->pointList.push_back({(float)x, (float)y, 1.0f});
        }break;
    }
}


int getPoints(const cv::Mat &calibImg, const std::string &filenameList3DPoints, std::vector<cv::Vec3f> &points2D, std::vector<cv::Vec4f> &points3D)
{
    WinInfo windowInfo = {
        calibImg.clone(),
        "Calibration image",
        {}
    };

    // show input image and install mouse callback
    cv::namedWindow( windowInfo.name.c_str(), cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
    cv::imshow( windowInfo.name.c_str(), windowInfo.img );
    cv::setMouseCallback(windowInfo.name.c_str(), getPointsCB, (void*) &windowInfo);
    // wait until any key was pressed
    cv::waitKey(0);
    
    cv::destroyWindow( windowInfo.name.c_str() );

    
    // allocate memory for point-lists (represented as matrix)
    points2D = windowInfo.pointList;
    points3D.resize(windowInfo.pointList.size());
    
    std::fstream calibFile(filenameList3DPoints.c_str(), std::ios::in);
    calibFile.exceptions(std::ifstream::failbit | std::ifstream::eofbit | std::ifstream::badbit);
    
    try {
        // read points, transform them into homogeneous coordinates
        for (unsigned i = 0; i < windowInfo.pointList.size(); i++) {
            // Read 3D point from file. Technically, they don't need to be on seperate lines,
            // they can just be x y z x y z x y z ...
            calibFile 
                >> points3D[i](0)
                >> points3D[i](1)
                >> points3D[i](2);

            points3D[i](3) = 1.0f;
        }
    } catch (const std::ios_base::failure &e) {
        std::cerr << "An error occured while reading the file " << filenameList3DPoints << ": " << e.what() << std::endl;
        if (calibFile.eof())
            std::cerr << "The file ended prematurely. Maybe less points specified than were clicked?" << std::endl;
        if (calibFile.fail())
            std::cerr << "The numbers in the file could not be parsed. Maybe accidental stray characters?" << std::endl;
        throw e;
    }

    return windowInfo.pointList.size();
}

std::ostream &operator<<(std::ostream &stream, const ProjectionMatrixInterpretation &info)
{
    stream 
        << "Principal distance:\t\t" << info.principalDistance << std::endl
        << "Skew:\t\t\t\t"           << info.skew << std::endl
        << "Aspect ratio:\t\t\t"     << info.aspectRatio << std::endl
        << "Principal point (x,y):\t\t["   << info.principalPoint[0] << ", " << info.principalPoint[1] << "]^T" << std::endl
        << std::endl 
        << "omega:\t"   << info.omega << std::endl
        << "phi:\t"     << info.phi << std::endl
        << "kappa:\t"   << info.kappa << std::endl
        << std::endl 
        << "external position (x,y,z):\t[" << info.cameraLocation[0] << ", " << info.cameraLocation[1] << ", " << info.cameraLocation[2] << "]^T" << std::endl;

    return stream;
}

}
