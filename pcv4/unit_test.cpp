
#include "Pcv4.h"

#include "Helper.h"

#include <opencv2/opencv.hpp>

#include <iostream>

#include <random>


// function calls processing functions
// output is tested on "correctness" 

using namespace pcv4;
using namespace cv;
using namespace std;

std::vector<cv::Vec3f> convertPointsOld2New(const Mat& pset) {
    std::vector<cv::Vec3f> result;
    result.resize(pset.cols);
    for (unsigned i = 0; i < pset.cols; i++)
        for (unsigned j = 0; j < 3; j++)
            result[i](j) = pset.at<float>(j, i);
    return result;
}


void getFakePoints(Mat& p_fst, Mat& p_snd){

    // allocate memory for point-lists (represented as matrix)
    p_fst = Mat(3, 8, CV_32FC1);
    p_snd = Mat(3, 8, CV_32FC1);
    
    p_fst.at<float>(0, 0) = 67;
    p_fst.at<float>(1, 0) = 18;
    p_fst.at<float>(2, 0) = 1;
    
    p_snd.at<float>(0, 0) = 5;
    p_snd.at<float>(1, 0) = 18;
    p_snd.at<float>(2, 0) = 1;

    
    p_fst.at<float>(0, 1) = 215;
    p_fst.at<float>(1, 1) = 22;
    p_fst.at<float>(2, 1) = 1;
    
    p_snd.at<float>(0, 1) = 161;
    p_snd.at<float>(1, 1) = 22;
    p_snd.at<float>(2, 1) = 1;


    p_fst.at<float>(0, 2) = 294;
    p_fst.at<float>(1, 2) = 74;
    p_fst.at<float>(2, 2) = 1;
    
    p_snd.at<float>(0, 2) = 227;
    p_snd.at<float>(1, 2) = 74;
    p_snd.at<float>(2, 2) = 1;
    
    
    p_fst.at<float>(0, 3) = 100;
    p_fst.at<float>(1, 3) = 85;
    p_fst.at<float>(2, 3) = 1;
    
    p_snd.at<float>(0, 3) = 41;
    p_snd.at<float>(1, 3) = 85;
    p_snd.at<float>(2, 3) = 1;

    
    p_fst.at<float>(0, 4) = 187;
    p_fst.at<float>(1, 4) = 115;
    p_fst.at<float>(2, 4) = 1;
    
    p_snd.at<float>(0, 4) = 100;
    p_snd.at<float>(1, 4) = 116;
    p_snd.at<float>(2, 4) = 1;


    p_fst.at<float>(0, 5) = 281;
    p_fst.at<float>(1, 5) = 153;
    p_fst.at<float>(2, 5) = 1;
    
    p_snd.at<float>(0, 5) = 220;
    p_snd.at<float>(1, 5) = 152;
    p_snd.at<float>(2, 5) = 1;

    
    p_fst.at<float>(0, 6) = 303;
    p_fst.at<float>(1, 6) = 194;
    p_fst.at<float>(2, 6) = 1;
    
    p_snd.at<float>(0, 6) = 237;
    p_snd.at<float>(1, 6) = 195;
    p_snd.at<float>(2, 6) = 1;

    
    p_fst.at<float>(0, 7) = 162;
    p_fst.at<float>(1, 7) = 225;
    p_fst.at<float>(2, 7) = 1;
    
    p_snd.at<float>(0, 7) = 74;
    p_snd.at<float>(1, 7) = 225;
    p_snd.at<float>(2, 7) = 1;
}


void getFakePointsWithOutliers(Mat& p_fst, Mat& p_snd){
    
    const unsigned numOutliers = 100;

    p_fst = Mat(3, 100+numOutliers, CV_32FC1);
    p_snd = Mat(3, 100+numOutliers, CV_32FC1);
    
    p_fst.at<float>(0, 0) = 314;
    p_fst.at<float>(1, 0) = 154;
    p_fst.at<float>(2, 0) = 1;
    p_snd.at<float>(0, 0) = 327;
    p_snd.at<float>(1, 0) = 148;
    p_snd.at<float>(2, 0) = 1;
    p_fst.at<float>(0, 1) = 86;
    p_fst.at<float>(1, 1) = 286;
    p_fst.at<float>(2, 1) = 1;
    p_snd.at<float>(0, 1) = 57.6;
    p_snd.at<float>(1, 1) = 288;
    p_snd.at<float>(2, 1) = 1;
    p_fst.at<float>(0, 2) = 346;
    p_fst.at<float>(1, 2) = 246;
    p_fst.at<float>(2, 2) = 1;
    p_snd.at<float>(0, 2) = 366;
    p_snd.at<float>(1, 2) = 246;
    p_snd.at<float>(2, 2) = 1;
    p_fst.at<float>(0, 3) = 91;
    p_fst.at<float>(1, 3) = 292;
    p_fst.at<float>(2, 3) = 1;
    p_snd.at<float>(0, 3) = 62.4;
    p_snd.at<float>(1, 3) = 295.2;
    p_snd.at<float>(2, 3) = 1;
    p_fst.at<float>(0, 4) = 330;
    p_fst.at<float>(1, 4) = 304;
    p_fst.at<float>(2, 4) = 1;
    p_snd.at<float>(0, 4) = 352.8;
    p_snd.at<float>(1, 4) = 306;
    p_snd.at<float>(2, 4) = 1;
    p_fst.at<float>(0, 5) = 332;
    p_fst.at<float>(1, 5) = 302;
    p_fst.at<float>(2, 5) = 1;
    p_snd.at<float>(0, 5) = 354;
    p_snd.at<float>(1, 5) = 303;
    p_snd.at<float>(2, 5) = 1;
    p_fst.at<float>(0, 6) = 97;
    p_fst.at<float>(1, 6) = 277;
    p_fst.at<float>(2, 6) = 1;
    p_snd.at<float>(0, 6) = 70.8;
    p_snd.at<float>(1, 6) = 278.4;
    p_snd.at<float>(2, 6) = 1;
    p_fst.at<float>(0, 7) = 303;
    p_fst.at<float>(1, 7) = 87;
    p_fst.at<float>(2, 7) = 1;
    p_snd.at<float>(0, 7) = 320.4;
    p_snd.at<float>(1, 7) = 79.2;
    p_snd.at<float>(2, 7) = 1;
    p_fst.at<float>(0, 8) = 97;
    p_fst.at<float>(1, 8) = 302;
    p_fst.at<float>(2, 8) = 1;
    p_snd.at<float>(0, 8) = 70.8;
    p_snd.at<float>(1, 8) = 306;
    p_snd.at<float>(2, 8) = 1;
    p_fst.at<float>(0, 9) = 96;
    p_fst.at<float>(1, 9) = 297;
    p_fst.at<float>(2, 9) = 1;
    p_snd.at<float>(0, 9) = 68.4;
    p_snd.at<float>(1, 9) = 301.2;
    p_snd.at<float>(2, 9) = 1;
    p_fst.at<float>(0, 10) = 356;
    p_fst.at<float>(1, 10) = 237;
    p_fst.at<float>(2, 10) = 1;
    p_snd.at<float>(0, 10) = 372;
    p_snd.at<float>(1, 10) = 236;
    p_snd.at<float>(2, 10) = 1;
    p_fst.at<float>(0, 11) = 367;
    p_fst.at<float>(1, 11) = 260;
    p_fst.at<float>(2, 11) = 1;
    p_snd.at<float>(0, 11) = 386;
    p_snd.at<float>(1, 11) = 260;
    p_snd.at<float>(2, 11) = 1;
    p_fst.at<float>(0, 12) = 304;
    p_fst.at<float>(1, 12) = 90;
    p_fst.at<float>(2, 12) = 1;
    p_snd.at<float>(0, 12) = 320.4;
    p_snd.at<float>(1, 12) = 82.8;
    p_snd.at<float>(2, 12) = 1;
    p_fst.at<float>(0, 13) = 343;
    p_fst.at<float>(1, 13) = 209;
    p_fst.at<float>(2, 13) = 1;
    p_snd.at<float>(0, 13) = 358;
    p_snd.at<float>(1, 13) = 207;
    p_snd.at<float>(2, 13) = 1;
    p_fst.at<float>(0, 14) = 326;
    p_fst.at<float>(1, 14) = 250;
    p_fst.at<float>(2, 14) = 1;
    p_snd.at<float>(0, 14) = 345.6;
    p_snd.at<float>(1, 14) = 249.12;
    p_snd.at<float>(2, 14) = 1;
    p_fst.at<float>(0, 15) = 86;
    p_fst.at<float>(1, 15) = 297;
    p_fst.at<float>(2, 15) = 1;
    p_snd.at<float>(0, 15) = 57.6;
    p_snd.at<float>(1, 15) = 301.2;
    p_snd.at<float>(2, 15) = 1;
    p_fst.at<float>(0, 16) = 341;
    p_fst.at<float>(1, 16) = 240;
    p_fst.at<float>(2, 16) = 1;
    p_snd.at<float>(0, 16) = 352.8;
    p_snd.at<float>(1, 16) = 238.8;
    p_snd.at<float>(2, 16) = 1;
    p_fst.at<float>(0, 17) = 93;
    p_fst.at<float>(1, 17) = 289;
    p_fst.at<float>(2, 17) = 1;
    p_snd.at<float>(0, 17) = 66.24;
    p_snd.at<float>(1, 17) = 292.32;
    p_snd.at<float>(2, 17) = 1;
    p_fst.at<float>(0, 18) = 338;
    p_fst.at<float>(1, 18) = 240;
    p_fst.at<float>(2, 18) = 1;
    p_snd.at<float>(0, 18) = 352;
    p_snd.at<float>(1, 18) = 239;
    p_snd.at<float>(2, 18) = 1;
    p_fst.at<float>(0, 19) = 94;
    p_fst.at<float>(1, 19) = 300;
    p_fst.at<float>(2, 19) = 1;
    p_snd.at<float>(0, 19) = 67.2;
    p_snd.at<float>(1, 19) = 303.6;
    p_snd.at<float>(2, 19) = 1;
    p_fst.at<float>(0, 20) = 366;
    p_fst.at<float>(1, 20) = 257;
    p_fst.at<float>(2, 20) = 1;
    p_snd.at<float>(0, 20) = 386;
    p_snd.at<float>(1, 20) = 257;
    p_snd.at<float>(2, 20) = 1;
    p_fst.at<float>(0, 21) = 337;
    p_fst.at<float>(1, 21) = 169;
    p_fst.at<float>(2, 21) = 1;
    p_snd.at<float>(0, 21) = 353;
    p_snd.at<float>(1, 21) = 166;
    p_snd.at<float>(2, 21) = 1;
    p_fst.at<float>(0, 22) = 96;
    p_fst.at<float>(1, 22) = 286;
    p_fst.at<float>(2, 22) = 1;
    p_snd.at<float>(0, 22) = 69.6;
    p_snd.at<float>(1, 22) = 288;
    p_snd.at<float>(2, 22) = 1;
    p_fst.at<float>(0, 23) = 332;
    p_fst.at<float>(1, 23) = 240;
    p_fst.at<float>(2, 23) = 1;
    p_snd.at<float>(0, 23) = 346.8;
    p_snd.at<float>(1, 23) = 240;
    p_snd.at<float>(2, 23) = 1;
    p_fst.at<float>(0, 24) = 351;
    p_fst.at<float>(1, 24) = 211;
    p_fst.at<float>(2, 24) = 1;
    p_snd.at<float>(0, 24) = 367;
    p_snd.at<float>(1, 24) = 209;
    p_snd.at<float>(2, 24) = 1;
    p_fst.at<float>(0, 25) = 369;
    p_fst.at<float>(1, 25) = 256;
    p_fst.at<float>(2, 25) = 1;
    p_snd.at<float>(0, 25) = 388;
    p_snd.at<float>(1, 25) = 256;
    p_snd.at<float>(2, 25) = 1;
    p_fst.at<float>(0, 26) = 85;
    p_fst.at<float>(1, 26) = 300;
    p_fst.at<float>(2, 26) = 1;
    p_snd.at<float>(0, 26) = 56.16;
    p_snd.at<float>(1, 26) = 303.84;
    p_snd.at<float>(2, 26) = 1;
    p_fst.at<float>(0, 27) = 342;
    p_fst.at<float>(1, 27) = 251;
    p_fst.at<float>(2, 27) = 1;
    p_snd.at<float>(0, 27) = 360;
    p_snd.at<float>(1, 27) = 251;
    p_snd.at<float>(2, 27) = 1;
    p_fst.at<float>(0, 28) = 346;
    p_fst.at<float>(1, 28) = 255;
    p_fst.at<float>(2, 28) = 1;
    p_snd.at<float>(0, 28) = 367;
    p_snd.at<float>(1, 28) = 255;
    p_snd.at<float>(2, 28) = 1;
    p_fst.at<float>(0, 29) = 354;
    p_fst.at<float>(1, 29) = 236;
    p_fst.at<float>(2, 29) = 1;
    p_snd.at<float>(0, 29) = 372;
    p_snd.at<float>(1, 29) = 236;
    p_snd.at<float>(2, 29) = 1;
    p_fst.at<float>(0, 30) = 352;
    p_fst.at<float>(1, 30) = 254;
    p_fst.at<float>(2, 30) = 1;
    p_snd.at<float>(0, 30) = 373;
    p_snd.at<float>(1, 30) = 254;
    p_snd.at<float>(2, 30) = 1;
    p_fst.at<float>(0, 31) = 727;
    p_fst.at<float>(1, 31) = 195;
    p_fst.at<float>(2, 31) = 1;
    p_snd.at<float>(0, 31) = 338.4;
    p_snd.at<float>(1, 31) = 138;
    p_snd.at<float>(2, 31) = 1;
    p_fst.at<float>(0, 32) = 360;
    p_fst.at<float>(1, 32) = 253;
    p_fst.at<float>(2, 32) = 1;
    p_snd.at<float>(0, 32) = 380;
    p_snd.at<float>(1, 32) = 253;
    p_snd.at<float>(2, 32) = 1;
    p_fst.at<float>(0, 33) = 331;
    p_fst.at<float>(1, 33) = 237;
    p_fst.at<float>(2, 33) = 1;
    p_snd.at<float>(0, 33) = 344;
    p_snd.at<float>(1, 33) = 236;
    p_snd.at<float>(2, 33) = 1;
    p_fst.at<float>(0, 34) = 85;
    p_fst.at<float>(1, 34) = 279;
    p_fst.at<float>(2, 34) = 1;
    p_snd.at<float>(0, 34) = 55.2;
    p_snd.at<float>(1, 34) = 280.8;
    p_snd.at<float>(2, 34) = 1;
    p_fst.at<float>(0, 35) = 336;
    p_fst.at<float>(1, 35) = 237;
    p_fst.at<float>(2, 35) = 1;
    p_snd.at<float>(0, 35) = 350;
    p_snd.at<float>(1, 35) = 236;
    p_snd.at<float>(2, 35) = 1;
    p_fst.at<float>(0, 36) = 341;
    p_fst.at<float>(1, 36) = 214;
    p_fst.at<float>(2, 36) = 1;
    p_snd.at<float>(0, 36) = 358;
    p_snd.at<float>(1, 36) = 212;
    p_snd.at<float>(2, 36) = 1;
    p_fst.at<float>(0, 37) = 86;
    p_fst.at<float>(1, 37) = 281;
    p_fst.at<float>(2, 37) = 1;
    p_snd.at<float>(0, 37) = 57.6;
    p_snd.at<float>(1, 37) = 283.2;
    p_snd.at<float>(2, 37) = 1;
    p_fst.at<float>(0, 38) = 335;
    p_fst.at<float>(1, 38) = 171;
    p_fst.at<float>(2, 38) = 1;
    p_snd.at<float>(0, 38) = 350;
    p_snd.at<float>(1, 38) = 167;
    p_snd.at<float>(2, 38) = 1;
    p_fst.at<float>(0, 39) = 344;
    p_fst.at<float>(1, 39) = 214;
    p_fst.at<float>(2, 39) = 1;
    p_snd.at<float>(0, 39) = 362;
    p_snd.at<float>(1, 39) = 212;
    p_snd.at<float>(2, 39) = 1;
    p_fst.at<float>(0, 40) = 94;
    p_fst.at<float>(1, 40) = 279;
    p_fst.at<float>(2, 40) = 1;
    p_snd.at<float>(0, 40) = 68;
    p_snd.at<float>(1, 40) = 280;
    p_snd.at<float>(2, 40) = 1;
    p_fst.at<float>(0, 41) = 420;
    p_fst.at<float>(1, 41) = 84;
    p_fst.at<float>(2, 41) = 1;
    p_snd.at<float>(0, 41) = 436.8;
    p_snd.at<float>(1, 41) = 82.8;
    p_snd.at<float>(2, 41) = 1;
    p_fst.at<float>(0, 42) = 499;
    p_fst.at<float>(1, 42) = 308;
    p_fst.at<float>(2, 42) = 1;
    p_snd.at<float>(0, 42) = 523;
    p_snd.at<float>(1, 42) = 308;
    p_snd.at<float>(2, 42) = 1;
    p_fst.at<float>(0, 43) = 374;
    p_fst.at<float>(1, 43) = 309;
    p_fst.at<float>(2, 43) = 1;
    p_snd.at<float>(0, 43) = 398;
    p_snd.at<float>(1, 43) = 310;
    p_snd.at<float>(2, 43) = 1;
    p_fst.at<float>(0, 44) = 393;
    p_fst.at<float>(1, 44) = 259;
    p_fst.at<float>(2, 44) = 1;
    p_snd.at<float>(0, 44) = 411;
    p_snd.at<float>(1, 44) = 259;
    p_snd.at<float>(2, 44) = 1;
    p_fst.at<float>(0, 45) = 328;
    p_fst.at<float>(1, 45) = 186;
    p_fst.at<float>(2, 45) = 1;
    p_snd.at<float>(0, 45) = 343;
    p_snd.at<float>(1, 45) = 183;
    p_snd.at<float>(2, 45) = 1;
    p_fst.at<float>(0, 46) = 354;
    p_fst.at<float>(1, 46) = 140;
    p_fst.at<float>(2, 46) = 1;
    p_snd.at<float>(0, 46) = 370;
    p_snd.at<float>(1, 46) = 136;
    p_snd.at<float>(2, 46) = 1;
    p_fst.at<float>(0, 47) = 345.6;
    p_fst.at<float>(1, 47) = 255.6;
    p_fst.at<float>(2, 47) = 1;
    p_snd.at<float>(0, 47) = 367;
    p_snd.at<float>(1, 47) = 255;
    p_snd.at<float>(2, 47) = 1;
    p_fst.at<float>(0, 48) = 315.6;
    p_fst.at<float>(1, 48) = 254.4;
    p_fst.at<float>(2, 48) = 1;
    p_snd.at<float>(0, 48) = 331.2;
    p_snd.at<float>(1, 48) = 253.44;
    p_snd.at<float>(2, 48) = 1;
    p_fst.at<float>(0, 49) = 360;
    p_fst.at<float>(1, 49) = 256.8;
    p_fst.at<float>(2, 49) = 1;
    p_snd.at<float>(0, 49) = 378.72;
    p_snd.at<float>(1, 49) = 256.32;
    p_snd.at<float>(2, 49) = 1;
    p_fst.at<float>(0, 50) = 350.4;
    p_fst.at<float>(1, 50) = 211.2;
    p_fst.at<float>(2, 50) = 1;
    p_snd.at<float>(0, 50) = 367;
    p_snd.at<float>(1, 50) = 209;
    p_snd.at<float>(2, 50) = 1;
    p_fst.at<float>(0, 51) = 327.6;
    p_fst.at<float>(1, 51) = 240;
    p_fst.at<float>(2, 51) = 1;
    p_snd.at<float>(0, 51) = 342;
    p_snd.at<float>(1, 51) = 238.8;
    p_snd.at<float>(2, 51) = 1;
    p_fst.at<float>(0, 52) = 330;
    p_fst.at<float>(1, 52) = 241.2;
    p_fst.at<float>(2, 52) = 1;
    p_snd.at<float>(0, 52) = 346.8;
    p_snd.at<float>(1, 52) = 240;
    p_snd.at<float>(2, 52) = 1;
    p_fst.at<float>(0, 53) = 337.2;
    p_fst.at<float>(1, 53) = 169.2;
    p_fst.at<float>(2, 53) = 1;
    p_snd.at<float>(0, 53) = 352.8;
    p_snd.at<float>(1, 53) = 165.6;
    p_snd.at<float>(2, 53) = 1;
    p_fst.at<float>(0, 54) = 345.6;
    p_fst.at<float>(1, 54) = 246;
    p_fst.at<float>(2, 54) = 1;
    p_snd.at<float>(0, 54) = 366;
    p_snd.at<float>(1, 54) = 246;
    p_snd.at<float>(2, 54) = 1;
    p_fst.at<float>(0, 55) = 326.4;
    p_fst.at<float>(1, 55) = 249.6;
    p_fst.at<float>(2, 55) = 1;
    p_snd.at<float>(0, 55) = 345.6;
    p_snd.at<float>(1, 55) = 249.12;
    p_snd.at<float>(2, 55) = 1;
    p_fst.at<float>(0, 56) = 342;
    p_fst.at<float>(1, 56) = 208.8;
    p_fst.at<float>(2, 56) = 1;
    p_snd.at<float>(0, 56) = 357.6;
    p_snd.at<float>(1, 56) = 206.4;
    p_snd.at<float>(2, 56) = 1;
    p_fst.at<float>(0, 57) = 360;
    p_fst.at<float>(1, 57) = 253.2;
    p_fst.at<float>(2, 57) = 1;
    p_snd.at<float>(0, 57) = 379.2;
    p_snd.at<float>(1, 57) = 253.2;
    p_snd.at<float>(2, 57) = 1;
    p_fst.at<float>(0, 58) = 332.4;
    p_fst.at<float>(1, 58) = 255.6;
    p_fst.at<float>(2, 58) = 1;
    p_snd.at<float>(0, 58) = 354.24;
    p_snd.at<float>(1, 58) = 254.88;
    p_snd.at<float>(2, 58) = 1;
    p_fst.at<float>(0, 59) = 366;
    p_fst.at<float>(1, 59) = 256.8;
    p_fst.at<float>(2, 59) = 1;
    p_snd.at<float>(0, 59) = 385.2;
    p_snd.at<float>(1, 59) = 256.8;
    p_snd.at<float>(2, 59) = 1;
    p_fst.at<float>(0, 60) = 368.4;
    p_fst.at<float>(1, 60) = 256.8;
    p_fst.at<float>(2, 60) = 1;
    p_snd.at<float>(0, 60) = 385.2;
    p_snd.at<float>(1, 60) = 256.8;
    p_snd.at<float>(2, 60) = 1;
    p_fst.at<float>(0, 61) = 313.2;
    p_fst.at<float>(1, 61) = 153.6;
    p_fst.at<float>(2, 61) = 1;
    p_snd.at<float>(0, 61) = 326.4;
    p_snd.at<float>(1, 61) = 148.8;
    p_snd.at<float>(2, 61) = 1;
    p_fst.at<float>(0, 62) = 346.8;
    p_fst.at<float>(1, 62) = 226.8;
    p_fst.at<float>(2, 62) = 1;
    p_snd.at<float>(0, 62) = 362.4;
    p_snd.at<float>(1, 62) = 225.6;
    p_snd.at<float>(2, 62) = 1;
    p_fst.at<float>(0, 63) = 304.8;
    p_fst.at<float>(1, 63) = 87.6;
    p_fst.at<float>(2, 63) = 1;
    p_snd.at<float>(0, 63) = 320.4;
    p_snd.at<float>(1, 63) = 79.2;
    p_snd.at<float>(2, 63) = 1;
    p_fst.at<float>(0, 64) = 319.2;
    p_fst.at<float>(1, 64) = 249.6;
    p_fst.at<float>(2, 64) = 1;
    p_snd.at<float>(0, 64) = 332;
    p_snd.at<float>(1, 64) = 249;
    p_snd.at<float>(2, 64) = 1;
    p_fst.at<float>(0, 65) = 331.2;
    p_fst.at<float>(1, 65) = 236.4;
    p_fst.at<float>(2, 65) = 1;
    p_snd.at<float>(0, 65) = 344.4;
    p_snd.at<float>(1, 65) = 235.2;
    p_snd.at<float>(2, 65) = 1;
    p_fst.at<float>(0, 66) = 339.6;
    p_fst.at<float>(1, 66) = 248.4;
    p_fst.at<float>(2, 66) = 1;
    p_snd.at<float>(0, 66) = 360;
    p_snd.at<float>(1, 66) = 247.68;
    p_snd.at<float>(2, 66) = 1;
    p_fst.at<float>(0, 67) = 97.2;
    p_fst.at<float>(1, 67) = 277.2;
    p_fst.at<float>(2, 67) = 1;
    p_snd.at<float>(0, 67) = 70.8;
    p_snd.at<float>(1, 67) = 278.4;
    p_snd.at<float>(2, 67) = 1;
    p_fst.at<float>(0, 68) = 85.2;
    p_fst.at<float>(1, 68) = 278.4;
    p_fst.at<float>(2, 68) = 1;
    p_snd.at<float>(0, 68) = 56.16;
    p_snd.at<float>(1, 68) = 279.36;
    p_snd.at<float>(2, 68) = 1;
    p_fst.at<float>(0, 69) = 86.4;
    p_fst.at<float>(1, 69) = 282;
    p_fst.at<float>(2, 69) = 1;
    p_snd.at<float>(0, 69) = 57.6;
    p_snd.at<float>(1, 69) = 283.68;
    p_snd.at<float>(2, 69) = 1;
    p_fst.at<float>(0, 70) = 96;
    p_fst.at<float>(1, 70) = 288;
    p_fst.at<float>(2, 70) = 1;
    p_snd.at<float>(0, 70) = 67.392;
    p_snd.at<float>(1, 70) = 290.304;
    p_snd.at<float>(2, 70) = 1;
    p_fst.at<float>(0, 71) = 93.6;
    p_fst.at<float>(1, 71) = 289.2;
    p_fst.at<float>(2, 71) = 1;
    p_snd.at<float>(0, 71) = 66.24;
    p_snd.at<float>(1, 71) = 292.32;
    p_snd.at<float>(2, 71) = 1;
    p_fst.at<float>(0, 72) = 94.8;
    p_fst.at<float>(1, 72) = 296.4;
    p_fst.at<float>(2, 72) = 1;
    p_snd.at<float>(0, 72) = 67.68;
    p_snd.at<float>(1, 72) = 300.96;
    p_snd.at<float>(2, 72) = 1;
    p_fst.at<float>(0, 73) = 86.4;
    p_fst.at<float>(1, 73) = 297.6;
    p_fst.at<float>(2, 73) = 1;
    p_snd.at<float>(0, 73) = 57.6;
    p_snd.at<float>(1, 73) = 300.96;
    p_snd.at<float>(2, 73) = 1;
    p_fst.at<float>(0, 74) = 98.4;
    p_fst.at<float>(1, 74) = 298.8;
    p_fst.at<float>(2, 74) = 1;
    p_snd.at<float>(0, 74) = 72;
    p_snd.at<float>(1, 74) = 302.4;
    p_snd.at<float>(2, 74) = 1;
    p_fst.at<float>(0, 75) = 88.8;
    p_fst.at<float>(1, 75) = 300;
    p_fst.at<float>(2, 75) = 1;
    p_snd.at<float>(0, 75) = 60.48;
    p_snd.at<float>(1, 75) = 303.84;
    p_snd.at<float>(2, 75) = 1;
    p_fst.at<float>(0, 76) = 93.6;
    p_fst.at<float>(1, 76) = 300;
    p_fst.at<float>(2, 76) = 1;
    p_snd.at<float>(0, 76) = 66.24;
    p_snd.at<float>(1, 76) = 303.84;
    p_snd.at<float>(2, 76) = 1;
    p_fst.at<float>(0, 77) = 344.4;
    p_fst.at<float>(1, 77) = 194.4;
    p_fst.at<float>(2, 77) = 1;
    p_snd.at<float>(0, 77) = 360;
    p_snd.at<float>(1, 77) = 192;
    p_snd.at<float>(2, 77) = 1;
    p_fst.at<float>(0, 78) = 327.6;
    p_fst.at<float>(1, 78) = 140.4;
    p_fst.at<float>(2, 78) = 1;
    p_snd.at<float>(0, 78) = 342;
    p_snd.at<float>(1, 78) = 135.6;
    p_snd.at<float>(2, 78) = 1;
    p_fst.at<float>(0, 79) = 338.4;
    p_fst.at<float>(1, 79) = 304.8;
    p_fst.at<float>(2, 79) = 1;
    p_snd.at<float>(0, 79) = 360;
    p_snd.at<float>(1, 79) = 306;
    p_snd.at<float>(2, 79) = 1;
    p_fst.at<float>(0, 80) = 426;
    p_fst.at<float>(1, 80) = 306;
    p_fst.at<float>(2, 80) = 1;
    p_snd.at<float>(0, 80) = 449.28;
    p_snd.at<float>(1, 80) = 306.72;
    p_snd.at<float>(2, 80) = 1;
    p_fst.at<float>(0, 81) = 338.4;
    p_fst.at<float>(1, 81) = 183.6;
    p_fst.at<float>(2, 81) = 1;
    p_snd.at<float>(0, 81) = 352.512;
    p_snd.at<float>(1, 81) = 179.712;
    p_snd.at<float>(2, 81) = 1;
    p_fst.at<float>(0, 82) = 378;
    p_fst.at<float>(1, 82) = 109.2;
    p_fst.at<float>(2, 82) = 1;
    p_snd.at<float>(0, 82) = 398.4;
    p_snd.at<float>(1, 82) = 106.8;
    p_snd.at<float>(2, 82) = 1;
    p_fst.at<float>(0, 83) = 332.4;
    p_fst.at<float>(1, 83) = 240;
    p_fst.at<float>(2, 83) = 1;
    p_snd.at<float>(0, 83) = 346.8;
    p_snd.at<float>(1, 83) = 240;
    p_snd.at<float>(2, 83) = 1;
    p_fst.at<float>(0, 84) = 346.8;
    p_fst.at<float>(1, 84) = 195.6;
    p_fst.at<float>(2, 84) = 1;
    p_snd.at<float>(0, 84) = 361;
    p_snd.at<float>(1, 84) = 193;
    p_snd.at<float>(2, 84) = 1;
    p_fst.at<float>(0, 85) = 340.8;
    p_fst.at<float>(1, 85) = 213.6;
    p_fst.at<float>(2, 85) = 1;
    p_snd.at<float>(0, 85) = 358;
    p_snd.at<float>(1, 85) = 212;
    p_snd.at<float>(2, 85) = 1;
    p_fst.at<float>(0, 86) = 303.6;
    p_fst.at<float>(1, 86) = 90;
    p_fst.at<float>(2, 86) = 1;
    p_snd.at<float>(0, 86) = 320.4;
    p_snd.at<float>(1, 86) = 82.8;
    p_snd.at<float>(2, 86) = 1;
    p_fst.at<float>(0, 87) = 334.8;
    p_fst.at<float>(1, 87) = 236.4;
    p_fst.at<float>(2, 87) = 1;
    p_snd.at<float>(0, 87) = 349.92;
    p_snd.at<float>(1, 87) = 234.72;
    p_snd.at<float>(2, 87) = 1;
    p_fst.at<float>(0, 88) = 331.2;
    p_fst.at<float>(1, 88) = 234;
    p_fst.at<float>(2, 88) = 1;
    p_snd.at<float>(0, 88) = 344.16;
    p_snd.at<float>(1, 88) = 233.28;
    p_snd.at<float>(2, 88) = 1;
    p_fst.at<float>(0, 89) = 320.4;
    p_fst.at<float>(1, 89) = 146.4;
    p_fst.at<float>(2, 89) = 1;
    p_snd.at<float>(0, 89) = 334.8;
    p_snd.at<float>(1, 89) = 141.6;
    p_snd.at<float>(2, 89) = 1;
    p_fst.at<float>(0, 90) = 324;
    p_fst.at<float>(1, 90) = 142.8;
    p_fst.at<float>(2, 90) = 1;
    p_snd.at<float>(0, 90) = 338.4;
    p_snd.at<float>(1, 90) = 138;
    p_snd.at<float>(2, 90) = 1;
    p_fst.at<float>(0, 91) = 499.2;
    p_fst.at<float>(1, 91) = 308.4;
    p_fst.at<float>(2, 91) = 1;
    p_snd.at<float>(0, 91) = 523.2;
    p_snd.at<float>(1, 91) = 308.4;
    p_snd.at<float>(2, 91) = 1;
    p_fst.at<float>(0, 92) = 718.56;
    p_fst.at<float>(1, 92) = 416.16;
    p_fst.at<float>(2, 92) = 1;
    p_snd.at<float>(0, 92) = 712.8;
    p_snd.at<float>(1, 92) = 406.8;
    p_snd.at<float>(2, 92) = 1;
    p_fst.at<float>(0, 93) = 335.52;
    p_fst.at<float>(1, 93) = 168.48;
    p_fst.at<float>(2, 93) = 1;
    p_snd.at<float>(0, 93) = 350.784;
    p_snd.at<float>(1, 93) = 164.16;
    p_snd.at<float>(2, 93) = 1;
    p_fst.at<float>(0, 94) = 718.56;
    p_fst.at<float>(1, 94) = 410.4;
    p_fst.at<float>(2, 94) = 1;
    p_snd.at<float>(0, 94) = 712.8;
    p_snd.at<float>(1, 94) = 400.8;
    p_snd.at<float>(2, 94) = 1;
    p_fst.at<float>(0, 95) = 498.24;
    p_fst.at<float>(1, 95) = 308.16;
    p_fst.at<float>(2, 95) = 1;
    p_snd.at<float>(0, 95) = 522.72;
    p_snd.at<float>(1, 95) = 308.16;
    p_snd.at<float>(2, 95) = 1;
    p_fst.at<float>(0, 96) = 345.6;
    p_fst.at<float>(1, 96) = 230.4;
    p_fst.at<float>(2, 96) = 1;
    p_snd.at<float>(0, 96) = 360;
    p_snd.at<float>(1, 96) = 229.2;
    p_snd.at<float>(2, 96) = 1;
    p_fst.at<float>(0, 97) = 96.48;
    p_fst.at<float>(1, 97) = 277.92;
    p_fst.at<float>(2, 97) = 1;
    p_snd.at<float>(0, 97) = 70.848;
    p_snd.at<float>(1, 97) = 278.208;
    p_snd.at<float>(2, 97) = 1;
    p_fst.at<float>(0, 98) = 367.2;
    p_fst.at<float>(1, 98) = 256.32;
    p_fst.at<float>(2, 98) = 1;
    p_snd.at<float>(0, 98) = 385.2;
    p_snd.at<float>(1, 98) = 256.8;
    p_snd.at<float>(2, 98) = 1;
    p_fst.at<float>(0, 99) = 90.72;
    p_fst.at<float>(1, 99) = 285.12;
    p_fst.at<float>(2, 99) = 1;
    p_snd.at<float>(0, 99) = 63.936;
    p_snd.at<float>(1, 99) = 286.848;
    p_snd.at<float>(2, 99) = 1;

    
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(0, 500);
    
    for (unsigned i = 0; i < numOutliers; i++) {
        unsigned idx1 = dist(rng);
        unsigned idx2 = dist(rng);
        p_fst.at<float>(0, 100+i) = dist(rng);
        p_fst.at<float>(1, 100+i) = dist(rng);
        p_fst.at<float>(2, 100+i) = 1.0f;
        p_snd.at<float>(0, 100+i) = dist(rng);
        p_snd.at<float>(1, 100+i) = dist(rng);
        p_snd.at<float>(2, 100+i) = 1.0f;
    }

}




bool checkDifference(float estimate, float trueValue, float maxDiff, const char *name, const char *functionName)
{
    if (std::abs(estimate - trueValue) > maxDiff) {
        std::cout 
            << "Warning: There seems to be a problem in " << functionName << " with the parameter " << name << ":" << std::endl
            << "\t==> Expected:" << trueValue << std::endl
            << "\t==> But got:" << estimate << std::endl;
        cin.get();
        return false;
    }
    return true;
}


bool test_MatSize(const string& casename, const Mat& a, int rows, int cols, int channels){
    if ( (a.rows != rows) || (a.cols != cols) || (a.channels() != channels) ) {
            cout << "\n"<<casename<<": fail\n" << "\nExpected:\n("<< rows<<","<<cols<<","<<channels << ")\nGiven:\n(" << a.rows<<"," << a.cols << "," << a.channels() <<")"<< endl;
            return false;  
    }else{
            return true;
    }
}

bool test_closeMat(const string& casename, const Mat& a,const Mat& b){
    float eps = pow(10,-3);
    if (sum(abs(a - b)).val[0] > eps){
        cout << "Wrong or inaccurate calculations!" << endl;
        if(casename.length() > 0){
            cout << "In matrix " << casename << "!" << endl;
        }
        cout << "\nExpected:\n"<< a << "\nGiven:\n" << b<<endl;
        return false;
    }
    return true;
}

bool test_getCondition2D(void){
    
    bool correct = true;

    cv::Mat p = (cv::Mat_<float>(3,6) <<  18.5, 99.1, 13.8, 242.1,
                                151.1, 243.1, 46.8, 146.5,
                                221.8, 52.5, 147.1, 224.5,
                                1, 1, 1, 1, 1, 1);
    cv::Matx33f Ttrue;
    Ttrue << 0.011883541, 0, -1.5204991, 0, 0.016626639, -2.3255126, 0, 0, 1;
        
    cv::Matx33f Test = getCondition2D(convertPointsOld2New(p));
    if (std::abs(Test(2,2)) < 1e-4f) {
        cout << "Warning: There seems to be a problem with getCondition2D(..)!" << endl;
        cout << "\t==> Expected T(2,2) to be nonzero!" << endl;
        return false;
    }
    Test *= 1.0f / Test(2,2);
    double eps = pow(10,-3);
    if (sum(abs(Mat(Test - Ttrue))).val[0] > eps){
        cout << "Warning: There seems to be a problem with getCondition2D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cout << "\t==> Expected:" << Ttrue << endl;
        cout << "\t==> But got:" << Test << endl;
        correct = false;
        cin.get();
    }
    return correct;
}


bool test_getFundamentalMatrix(void){
    cout << "===============================\nPcv4::getFundamentalMatrix(..):\n"<<flush;
    try{
        bool correct = true;
        
        Matx33f Ftrue;
        Ftrue <<  6.4590546e-07, -0.00014758465, 0.015314385,
                0.00015971341, -2.0858946e-05, -0.039460059,
                -0.016546328, 0.031596929, 1;
        
        Mat p_fst, p_snd;
        getFakePoints(p_fst, p_snd);
        Matx33f F = getFundamentalMatrix(convertPointsOld2New(p_fst), convertPointsOld2New(p_snd));
        if (std::abs(F(2,2)) < 1e-4f) {
            cout << "Warning: There seems to be a problem with getFundamentalMatrix(..)!" << endl;
            cout << "\t==> Expected F(2,2) to be nonzero!" << endl;
            return false;
        }
        F *= 1.0f / F(2,2);
        Ftrue *= 1.0f / Ftrue(2,2);
        correct = correct && test_closeMat("", Mat(Ftrue), Mat(F));
        return correct;
    }
    catch(const std::exception &exc){
        cout << exc.what();
        return false;
    }
}



bool test_getDesignMatrix_fundamental(void){
    cout << "===============================\nPcv4::getDesignMatrix_fundamental(..):\n"<<flush;
    try{
        bool correct = true;
    
        Mat p1 = (Mat_<float>(3,8) << -1.8596188, 0.19237423, 1.2876947, -1.4020798, -0.1958406, 1.1074524, 1.4124782, -0.54246116,
                                    -1.5204918, -1.4549181, -0.60245907, -0.4221313, 0.069671988, 0.69262278, 1.3647538, 1.8729507,
                                    1, 1, 1, 1, 1, 1, 1, 1);
        Mat p2 = (Mat_<float>(3,8) << -1.64, 0.35679984, 1.2015998, -1.1791999, -0.42400002, 1.112, 1.3295999, -0.7568,
                                    -1.5194274, -1.4539878, -0.60327208, -0.4233129, 0.083844543, 0.67280149, 1.3762779, 1.8670754,
                                    1, 1, 1, 1, 1, 1, 1, 1);
    
        Mat_<float> Fest = getDesignMatrix_fundamental(convertPointsOld2New(p1), convertPointsOld2New(p2));
        correct = (test_MatSize("Wrong dimensions!", Fest, 8, 9, 1) 
        || test_MatSize("Wrong dimensions!", Fest, 9, 9, 1) )
        && correct;

        Mat Ftrue;
        if (Fest.rows == 8){
        Ftrue = (Mat_<float>(8,9) << 3.0497749, 2.4936066, -1.64, 2.8255558, 2.310277, -1.5194274, -1.8596188, -1.5204918, 1,
                                        0.068639092, -0.51911455, 0.35679984, -0.27970979, 2.1154332, -1.4539878, 0.19237423, -1.4549181, 1,
                                        1.5472938, -0.72391474, 1.2015998, -0.77683026, 0.36344674, -0.60327208, 1.2876947, -0.60245907, 1,
                                        1.6533325, 0.49777719, -1.1791999, 0.5935185, 0.17869362, -0.4233129, -1.4020798, -0.4221313, 1,
                                        0.083036415, -0.029540924, -0.42400002, -0.016420165, 0.0058416161, 0.083844543, -0.1958406, 0.069671988, 1,
                                        1.231487, 0.7701965, 1.112, 0.74509561, 0.46599764, 0.67280149, 1.1074524, 0.69262278, 1,
                                        1.8780308, 1.8145765, 1.3295999, 1.9439626, 1.8782806, 1.3762779, 1.4124782, 1.3647538, 1,
                                        0.41053459, -1.4174491, -0.7568, -1.012816, 3.4969401, 1.8670754, -0.54246116, 1.8729507, 1);
        }else{
        Ftrue = (Mat_<float>(9,9) << 3.0497749, 2.4936066, -1.64, 2.8255558, 2.310277, -1.5194274, -1.8596188, -1.5204918, 1,
                                        0.068639092, -0.51911455, 0.35679984, -0.27970979, 2.1154332, -1.4539878, 0.19237423, -1.4549181, 1,
                                        1.5472938, -0.72391474, 1.2015998, -0.77683026, 0.36344674, -0.60327208, 1.2876947, -0.60245907, 1,
                                        1.6533325, 0.49777719, -1.1791999, 0.5935185, 0.17869362, -0.4233129, -1.4020798, -0.4221313, 1,
                                        0.083036415, -0.029540924, -0.42400002, -0.016420165, 0.0058416161, 0.083844543, -0.1958406, 0.069671988, 1,
                                        1.231487, 0.7701965, 1.112, 0.74509561, 0.46599764, 0.67280149, 1.1074524, 0.69262278, 1,
                                        1.8780308, 1.8145765, 1.3295999, 1.9439626, 1.8782806, 1.3762779, 1.4124782, 1.3647538, 1,
                                        0.41053459, -1.4174491, -0.7568, -1.012816, 3.4969401, 1.8670754, -0.54246116, 1.8729507, 1,
                                        0,0,0,0,0,0,0,0,0);
        }                                
        correct = test_closeMat("", Ftrue, Fest) && correct;
        return correct;
    }
    catch(const std::exception &exc){
        cout <<  exc.what();
        return false;
    }
}

bool test_solve_dlt(void){
    cout << "===============================\nPcv4::solve_dlt_fundamental(..):\n"<<flush;
    try{
        bool correct = true;
    
        Mat_<float> A(8, 9);
        A << 

        0.55613172, 0.36151901, -0.11716299, 0.42143318, 0.50170219, -0.081145406, -0.14141218, -0.081358112, 0.28945163,
            0.037880164, -0.20165631, 0.27597162, -0.17037115, 0.45962757, 0.49632433, 0.33699739, 0.49706647, 0.18567577,
            -0.15503874, -0.42786688, -0.46612, -0.3501814, 0.46457446, -0.081916817, -0.46723419, -0.081672095, 0.074027866,
            0.13710688, -0.26806444, 0.39586496, -0.27092993, 0.16518487, -0.46980408, 0.37809196, -0.46927336, 0.2608797,
            0.59673834, -0.2240283, -0.15800957, -0.28392494, -0.5223974, 0.13491097, -0.12557232, 0.13693732, 0.40313032,
            0.53824687, -0.15117355, 0.070677012, -0.15590209, 0.12148597, -0.008281284, 0.0014170756, -0.00044638285, -0.80206388,
            -0.021527633, -0.058972765, 0.70984244, 0.040031876, -0.014713437, -0.0036024868, -0.69528753, 0.067997277, 0.047980171,
            -0.010421497, 0.45400813, -0.008048675, -0.44800973, 0.050085664, -0.54432148, -0.016960399, 0.54209197, 0.0066823587,
            0.0083019603, -0.53950614, -0.047245972, 0.53861266, -0.059489254, -0.45286086, 0.075440452, 0.44964278, -0.0060508098;
        /*
        3.0497749, 2.4936066, -1.64, 2.8255558, 2.310277, -1.5194274, -1.8596188, -1.5204918, 1,
        0.068639092, -0.51911455, 0.35679984, -0.27970979, 2.1154332, -1.4539878, 0.19237423, -1.4549181, 1,
        1.5472938, -0.72391474, 1.2015998, -0.77683026, 0.36344674, -0.60327208, 1.2876947, -0.60245907, 1,
        1.6533325, 0.49777719, -1.1791999, 0.5935185, 0.17869362, -0.4233129, -1.4020798, -0.4221313, 1,
        0.083036415, -0.029540924, -0.42400002, -0.016420165, 0.0058416161, 0.083844543, -0.1958406, 0.069671988, 1,
        1.231487, 0.7701965, 1.112, 0.74509561, 0.46599764, 0.67280149, 1.1074524, 0.69262278, 1,
        1.8780308, 1.8145765, 1.3295999, 1.9439626, 1.8782806, 1.3762779, 1.4124782, 1.3647538, 1,
        0.41053459, -1.4174491, -0.7568, -1.012816, 3.4969401, 1.8670754, -0.54246116, 1.8729507, 1; */
        Matx33f Fest = solve_dlt_fundamental(A);
        if (std::abs(Fest(2,2)) < 1e-4f) {
            cout << "Warning: There seems to be a problem with solve_dlt_fundamental(..)!" << endl;
            cout << "\t==> Expected F(2,2) to be nonzero!" << endl;
            cout << Fest << endl;
            return false;
        }
        Fest *= 1.0f / Fest(2,2);
        Matx33f Ftrue;
        Ftrue  << 0.0083019603, -0.53950614, -0.047245972,
                                    0.53861266, -0.059489254, -0.45286086,
                                    0.075440452, 0.44964278, -0.0060508098;

        /*        -0, -472.06815, -60.662292,
                477.03021, -52.180977, -868.94147,
                84.680382, 863.7431, 1; */
        Ftrue *= 1.0f / Ftrue(2,2);
        correct = test_closeMat("", Mat(Ftrue), Mat(Fest)) && correct;
        return correct;
    }
    catch(const std::exception &exc){
        cout <<  exc.what();
        return false;
    }
}

bool test_decondition(void){
    cout << "===============================\nPcv4::decondition(..):\n"<<flush;
    try{
        bool correct = true;
        Matx33f H;
        H << 0.0027884692, -0.53886771, -0.053913236,
                                    0.53946984, -0.059588462, -0.45182425,
                                    0.068957359, 0.45039368, -0.01389052;
        Matx33f T1;
        T1 << 0.013864818, 0, -2.7885616, 0, 0.016393442, -1.8155738, 0, 0, 1;
        Matx33f T2;
        T2 << 0.0128, 0, -1.704, 0, 0.016359918, -1.813906, 0, 0, 1;
        H = decondition_fundamental(T1, T2, H);
        if (std::abs(H(2,2)) < 1e-4f) {
            cout << "Warning: There seems to be a problem with decondition_fundamental(..)!" << endl;
            cout << "\t==> Expected F(2,2) to be nonzero!" << endl;
            return false;
        }
        H *= 1.0f / H(2,2);
        Matx33f Htrue;
        Htrue << 6.4590546e-07, -0.00014758465, 0.015314385, 0.00015971341, -2.0858946e-05, -0.039460059, -0.016546328, 0.031596929, 1;
        correct = test_closeMat("", Mat(Htrue), Mat(H)) && correct;
        return correct;
    }
    catch(const std::exception &exc){
        cout <<  exc.what();
        return false;
    }
}


bool test_forceSingularity(void){
    cout << "===============================\nPcv4::forceSingularity(..):\n"<<flush;
    try{
        bool correct = true;
    
        Matx33f Fsest;
        Fsest << 0.0083019603, -0.53950614, -0.047245972,
                    0.53861266, -0.059489254, -0.45286086,
                0.075440452, 0.44964278, -0.0060508098;
                
        Fsest = forceSingularity(Fsest);
        Matx33f Fstrue;
        Fstrue  << 0.0027884692, -0.53886771, -0.053913236,
                    0.53946984, -0.059588462, -0.45182425,
                    0.068957359, 0.45039368, -0.01389052;
        correct = test_closeMat("", Mat(Fstrue), Mat(Fsest)) && correct;
        return correct;
    }
    catch(const std::exception &exc){
        cout <<  exc.what();
        return false;
    }
}

bool test_getError(void)
{
    cout << "===============================\nPcv4::getError(..):\n"<<flush;
    try{
        bool correct = true;
        
        Mat Ftrue = (Mat_<float>(3,3) <<  0.18009815, 0.84612828, -124.47226,
                                        0.51897198, 0.75658411, -182.07408,
                                        0.00088265416, 0.0073684035, -0.94836563);
        double erTrue = 3983.8915033623125;

        Mat p_fst, p_snd;
        getFakePoints(p_fst, p_snd);
        double erEst = getError(convertPointsOld2New(p_fst), convertPointsOld2New(p_snd), Ftrue);
        
        float eps = pow(10,-3);
        if (abs(erEst - erTrue) > eps){
            cout << "Wrong or inaccurate calculations!" << endl;
            cout << "In value \"Error\"!" << endl;
            cout << "\nExpected:\n"<< erTrue << "\nGiven:\n" << erEst <<endl;
            return false;
        }
        return correct;
    }
    catch(const std::exception &exc){
        cout <<  exc.what();
        return false;
    }
}

bool test_countInliers(void)
{
    cout << "===============================\nPcv4::countInliers(..):\n"<<flush;
    try{
        bool correct = true;
        
        Mat F = (Mat_<float>(3,3) <<  6.4590546e-07, -0.00014758465, 0.015314385,
                                        0.00015971341, -2.0858946e-05, -0.039460059,
                                        -0.016546328, 0.031596929, 1);


        Mat p_fst, p_snd;
        getFakePoints(p_fst, p_snd);
        unsigned numInliers = countInliers(convertPointsOld2New(p_fst), convertPointsOld2New(p_snd), F, 1.0f);
        unsigned true_numInliers = 5;

        if (numInliers != true_numInliers){
            cout << "Wrong or inaccurate calculations!" << endl;
            cout << "In value \"Error\"!" << endl;
            cout << "\nExpected:\n"<< true_numInliers << "\nGiven:\n" << numInliers <<endl;
            return false;
        }

        return correct;
    }
    catch(const std::exception &exc){
        cout <<  exc.what();
        return false;
    }
}

bool test_RANSAC(void)
{
    cout << "===============================\nPcv4::estimateFundamentalRANSAC(..):\n"<<flush;
    try{
        bool correct = true;

        Mat p_fst, p_snd;
        getFakePointsWithOutliers(p_fst, p_snd);
        cv::Matx33f F = estimateFundamentalRANSAC(convertPointsOld2New(p_fst), convertPointsOld2New(p_snd), 2000, 1.0f);
        unsigned numInliers = countInliers(convertPointsOld2New(p_fst), convertPointsOld2New(p_snd), F, 1.0f);
        if (numInliers < 70) {
            std::cout << "The solution that RANSAC finds is not very good (has few inliers)" << std::endl;
            std::cout << "Got " << numInliers << " inliers but expected around 100" << std::endl;
            correct = false;
        }
        if (numInliers > 150) {
            std::cout << "Something weird is going on with the test" << std::endl;
            correct = false;
        }
        
        return correct;

    }
    catch(const std::exception &exc){
        cout <<  exc.what();
        return false;
    }
}
        

int main() {

    cout << endl << "********************" << endl;
    cout << "Testing: Start" << endl;
   
    
    bool correct = true;
    correct &= test_getFundamentalMatrix();
    correct &= test_getCondition2D();
    correct &= test_getDesignMatrix_fundamental();
    correct &= test_solve_dlt();
    correct &= test_decondition();
    correct &= test_forceSingularity();
    correct &= test_getError();
    correct &= test_countInliers();
    correct &= test_RANSAC();

    cout << "Testing: Done" << endl;
    if (correct)
        cout << "Everything seems (!) to be correct." << endl;
    else
        cout << "There seem to be problems." << endl;
    cout << endl << "********************" << endl << endl;

    
    cout << "Press enter to continue..." << endl;
    cin.get();

    return 0;
}
