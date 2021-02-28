//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Pcv5.h"


#include "Helper.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <fstream>


using namespace std;
using namespace pcv5;


void writeStateToPly(const std::string &filename, BundleAdjustment::BAState &state)
{
    
    unsigned numVerticesTracks = state.m_tracks.size() * 4;
    unsigned numVerticesCameras = state.m_cameras.size() * 5;
    
    unsigned numVertices = numVerticesTracks + numVerticesCameras;
    
    unsigned numTriangles =
            state.m_tracks.size() * 4 +
            state.m_cameras.size() * 4;
    
    std::fstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    file.open(filename, std::fstream::out);

    file
        << "ply" << std::endl
        << "format ascii 1.0" << std::endl
        << "element vertex " << numVertices << std::endl
        << "property float x" << std::endl
        << "property float y" << std::endl
        << "property float z" << std::endl
        << "property uchar red" << std::endl
        << "property uchar green" << std::endl
        << "property uchar blue" << std::endl
        << "element face " << numTriangles << std::endl
        << "property list uchar int vertex_indices" << std::endl
        << "end_header" << std::endl;

        
    for (unsigned i = 0; i < state.m_tracks.size(); i++) {
        
        cv::Vec3f center = hom2eucl(state.m_tracks[i].location);
        
        for (unsigned j = 0; j < 3; j++)
            center(j) = std::min(std::max(center(j), -50.0f), 50.0f); // clamp
        
        std::string color = "255 255 255";
        float size = 0.01f;
        float s2 = size * std::sqrt(1.0f/2.0f);
        
        file
            << center(0)-size << ' ' << center(1)      << ' ' << center(2)-s2 << ' ' << color << std::endl
            << center(0)+size << ' ' << center(1)      << ' ' << center(2)-s2 << ' ' << color << std::endl
            << center(0)      << ' ' << center(1)-size << ' ' << center(2)+s2 << ' ' << color << std::endl
            << center(0)      << ' ' << center(1)+size << ' ' << center(2)+s2 << ' ' << color << std::endl;
    }
    
    for (unsigned i = 0; i < state.m_cameras.size(); i++) {
        std::string color = "255 0 0";
        
        const cv::Matx44f &H = state.m_cameras[i].H;
        
        std::cout << "camera " << i << " " << H << std::endl;
        
        cv::Matx44f Hinv = H.inv();
        
        float size = 0.2f;
        
        cv::Vec4f corners[5];
        corners[0] = Hinv * cv::Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
        corners[1] = Hinv * cv::Vec4f(+size, +size, size, 1.0f);
        corners[2] = Hinv * cv::Vec4f(-size, +size, size, 1.0f);
        corners[3] = Hinv * cv::Vec4f(+size, -size, size, 1.0f);
        corners[4] = Hinv * cv::Vec4f(-size, -size, size, 1.0f);
        
        for (unsigned j = 0; j < 5; j++) {
            corners[j] /= corners[j](3);
            file << corners[j](0) << ' ' << corners[j](1) << ' ' << corners[j](2) << ' ' << color << std::endl;
        }
        
    }
    
    for (unsigned i = 0; i < state.m_tracks.size(); i++) {
        file 
            << "3 " << i*4 + 0 << ' ' << i*4 + 1 << ' ' << i*4 + 3 << std::endl
            << "3 " << i*4 + 1 << ' ' << i*4 + 2 << ' ' << i*4 + 3 << std::endl
            << "3 " << i*4 + 2 << ' ' << i*4 + 0 << ' ' << i*4 + 3 << std::endl
            << "3 " << i*4 + 0 << ' ' << i*4 + 1 << ' ' << i*4 + 2 << std::endl;
    }

    for (unsigned i = 0; i < state.m_cameras.size(); i++) {
        unsigned o = numVerticesTracks + i*5;
        file 
            << "3 " << o + 0 << ' ' << o + 1 << ' ' << o + 2 << std::endl
            << "3 " << o + 0 << ' ' << o + 2 << ' ' << o + 4 << std::endl
            << "3 " << o + 0 << ' ' << o + 4 << ' ' << o + 3 << std::endl
            << "3 " << o + 0 << ' ' << o + 3 << ' ' << o + 1 << std::endl;
    }

}



// usage: path to image in argv[1]
// main function. loads and saves image
int main(int argc, char** argv) {

    // check if image paths were defined
    if (argc < 5){
        cerr << "Usage: main <focal length> <principal point X> <principal point y> <path to 1st image> <path to 2nd image> <path to 3rd image> ..." << endl;
        cerr << "Press enter to continue..." << endl;
        cin.get();
        return -1;
    }

    std::vector<std::string> imagesFilenames;
    imagesFilenames.reserve(argc-1);
    for (unsigned i = 4; i < argc; i++)
        imagesFilenames.push_back(argv[i]);
    
    cv::Matx33f K = cv::Matx33f::eye();

    K(0, 0) = 
    K(1, 1) = std::stoi(argv[1]);
    K(0, 2) = std::stoi(argv[2]);
    K(1, 2) = std::stoi(argv[3]);

    

    Scene scene = buildScene(imagesFilenames);
    std::unique_ptr<BundleAdjustment::BAState> state(new BundleAdjustment::BAState(scene));
    produceInitialState(scene, K, *state);
    
    BundleAdjustment bundleAdjustment(scene);
    
    LevenbergMarquardt lm(bundleAdjustment, std::move(state));
    
    BundleAdjustment::BAState *initialState = (BundleAdjustment::BAState *) lm.getState();
    writeStateToPly("beforeBA.ply", *initialState);

    std::cout << "Initial internal calibration: " << std::endl;
    std::cout << initialState->m_internalCalibs[0].K(0, 0) << " \t"
              << initialState->m_internalCalibs[0].K(0, 1) << " \t"
              << initialState->m_internalCalibs[0].K(0, 2) << std::endl;
    std::cout << initialState->m_internalCalibs[0].K(1, 0) << " \t"
              << initialState->m_internalCalibs[0].K(1, 1) << " \t"
              << initialState->m_internalCalibs[0].K(1, 2) << std::endl;
    std::cout << initialState->m_internalCalibs[0].K(2, 0) << " \t"
              << initialState->m_internalCalibs[0].K(2, 1) << " \t"
              << initialState->m_internalCalibs[0].K(2, 2) << std::endl;
    
    std::cout << "Initial camera poses calibration: " << std::endl;
    for (unsigned i = 0; i < initialState->m_cameras.size(); i++) {
        std::cout << "  camera " << i << std::endl;
        for (unsigned j = 0; j < 3; j++) 
            std::cout << "   "  << initialState->m_cameras[i].H(j, 0) << " \t"
                                << initialState->m_cameras[i].H(j, 1) << " \t"
                                << initialState->m_cameras[i].H(j, 2) << " \t | "
                                << initialState->m_cameras[i].H(j, 3) << std::endl;
        
    }
    
    for (unsigned i = 0; i < 500; i++) {
        lm.iterate();
        
        float sumWeights = 0.0f;
        for (const auto &c : scene.cameras)
            for (const auto &kp : c.keypoints)
                sumWeights += kp.weight*kp.weight;

        std::cout << "iter " << i << " error: " << std::sqrt(lm.getLastError() / sumWeights) << " (reprojection stddev in pixels)" << std::endl;
        
        if (i % 10 == 9)
            bundleAdjustment.downweightOutlierKeypoints((BundleAdjustment::BAState&) *lm.getState());
        
        if (lm.getDamping() > 1e6f) break;
    }
    
    BundleAdjustment::BAState *finalState = (BundleAdjustment::BAState *) lm.getState();
    writeStateToPly("afterBA.ply", *finalState);
    
    
    
    std::cout << "Final internal calibration: " << std::endl;
    std::cout << finalState->m_internalCalibs[0].K(0, 0) << " \t"
              << finalState->m_internalCalibs[0].K(0, 1) << " \t"
              << finalState->m_internalCalibs[0].K(0, 2) << std::endl;
    std::cout << finalState->m_internalCalibs[0].K(1, 0) << " \t"
              << finalState->m_internalCalibs[0].K(1, 1) << " \t"
              << finalState->m_internalCalibs[0].K(1, 2) << std::endl;
    std::cout << finalState->m_internalCalibs[0].K(2, 0) << " \t"
              << finalState->m_internalCalibs[0].K(2, 1) << " \t"
              << finalState->m_internalCalibs[0].K(2, 2) << std::endl;

    std::cout << "Final camera poses calibration: " << std::endl;
    for (unsigned i = 0; i < finalState->m_cameras.size(); i++) {
        std::cout << "  camera " << i << std::endl;
        for (unsigned j = 0; j < 3; j++) 
            std::cout << "   "  << finalState->m_cameras[i].H(j, 0) << " \t"
                                << finalState->m_cameras[i].H(j, 1) << " \t"
                                << finalState->m_cameras[i].H(j, 2) << " \t | "
                                << finalState->m_cameras[i].H(j, 3) << std::endl;
        
    }

              
    cout << "Press enter to continue..." << endl;
    cin.get();

    return 0;

}
