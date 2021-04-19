//============================================================================
// Name        : main.cpp
// Author      : Irina Nurutdinova
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Pcv1.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;
using namespace std;
using namespace pcv1;


void test_getConnectingLine(void){

    cv::Vec3f v1(0, 0, 1);
    cv::Vec3f v2(1, 1, 1);
    cv::Vec3f lt(-1, 1, 0);
    cv::Vec3f lc = getConnectingLine_2D(v1, v2);
    
    if (sum(lc != lt).val[0] != 0){
        cout << "There seems to be a problem with getConnectingLine_2D(..)!" << endl;
    }
}

void test_getScaleMatrix(void){

    cv::Matx33f St(3, 0, 0, 0, 3, 0, 0, 0, 1);
    cv::Matx33f Sc = getScalingMatrix_2D(3);
    
    if (sum(Sc != St).val[0] != 0){
        cout << "There seems to be a problem with getScalingMatrix_2D(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }	
}

void test_getRotMatrix(void){

    cv::Matx33f Rt(1./sqrt(2), -1./sqrt(2), 0, 1./sqrt(2), 1./sqrt(2), 0, 0, 0, 1);
    cv::Matx33f Rc = getRotationMatrix_2D(45);
    
    if (sum(Rc != Rt).val[0] != 0){
        cout << "There seems to be a problem with getRotationMatrix_2D(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }	
}

void test_getTranslMatrix(void){

    cv::Matx33f Tt(1, 0, -1, 0, 1, -1, 0, 0, 1);
    cv::Matx33f Tc = getTranslationMatrix_2D(-1,-1);
    
    if (sum(Tc != Tt).val[0] != 0){
        cout << "There seems to be a problem with getTranslationMatrix_2D(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }	
}

void test_getH(void){

    cv::Matx33f St(3, 0, 0, 0, 3, 0, 0, 0, 1);
    cv::Matx33f Rt(0, -1, 0, 1, 0, 0, 0, 0, 1);
    cv::Matx33f Tt(1, 0, -1, 0, 1, -1, 0, 0, 1);
    cv::Matx33f Ht(0, -3, 3, 3, 0, -3, 0, 0, 1);
    cv::Matx33f Hc = getH_2D(Tt, Rt, St);

    if (sum(Hc != Ht).val[0] != 0){
        cout << "There seems to be a problem with getH_2D(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }
}

void test_applyH(void){
    cv::Matx33f H(0, -3, 3, 
                  3, 0, -3, 
                  0, 0, 1);

    {
        std::vector<cv::Vec3f> v;
        std::vector<cv::Vec3f> vnt;
        for (unsigned i = 0; i < 10; i++) {
            v.push_back({
                1.0f*i, 1.0f, 1.0f
            });
            vnt.push_back({
                0.0f, 3.0f * i - 3.0f, 1.0f
            });
        }

        auto vnc = applyH_2D(v, H, GEOM_TYPE_POINT);

        if (v.size() != vnc.size()) {
            cout << "There seems to be a problem with applyH_2D(..) for points! The number of returned points does not match the number of given points." << endl;
            cout << "Press enter to continue..." << endl;
            cin.get();
            exit(-1);
        }

        for (unsigned i = 0; i < v.size(); i++) {
            if (sum(vnc[i] != vnt[i]).val[0] != 0) {
                cout << "There seems to be a problem with applyH_2D(..) for points for the " << i << "th element in the std::vector!" << endl;
                cout << vnc[i] << endl << endl << vnt[i] << endl;
                cout << "Press enter to continue..." << endl;
                cin.get();
                exit(-1);
            }
        }
    }

    {
        std::vector<cv::Vec3f> v;
        std::vector<cv::Vec3f> vnt;
        for (unsigned i = 0; i < 10; i++) {
            v.push_back({
                -1.0f*i, 1.0f, 0.0f
            });
            vnt.push_back({
                -1.0f/3.0f, -1.0f/3.0f*i, 1.0f - i
            });
        }

        auto vnc = applyH_2D(v, H, GEOM_TYPE_LINE);

        if (v.size() != vnc.size()) {
            cout << "There seems to be a problem with applyH_2D(..) for lines! The number of returned lines does not match the number of given lines." << endl;
            cout << "Press enter to continue..." << endl;
            cin.get();
            exit(-1);
        }

        for (unsigned i = 0; i < v.size(); i++) {
            if (sum(vnc[i] != vnt[i]).val[0] != 0){
                cout << "There seems to be a problem with applyH_2D(..) for lines for the " << i << "th element in the std::vector!" << endl;
                cout << "Press enter to continue..." << endl;
                cin.get();
                exit(-1);
            }
        }
    }
}

void test_isPointOnLine(void){	
    try{
    
        Mat v = (Mat_<float>(3,1) << 1, 1, 1);
        Mat l = (Mat_<float>(3,1) << -1, 1, 0);

        bool t = true;
        bool c = isPointOnLine_2D(Vec3f(v), Vec3f(l));
        
        if (t != c){
            cout << "isPointOnLine_2D: fail" << "\nExpected:\n"<< t << "\nGiven:\n" << c<<endl; 
            cout << "Press enter to continue..." << endl;
            cin.get();
            exit(-1);
        }
        
        v = (Mat_<float>(3,1) << 0, 1, 1);
        l = (Mat_<float>(3,1) << -1, 1, 1);

        t = false;
        c = isPointOnLine_2D(Vec3f(v), Vec3f(l));
        if (t != c){
            cout << "isPointOnLine_2D: fail" << "\nExpected:\n"<< t << "\nGiven:\n" << c<<endl; 
            cout << "Press enter to continue..." << endl;
            cin.get();
            exit(-1);
        }
        
        v = (Mat_<float>(3,1) << 0, 1, 1);
        l = (Mat_<float>(3,1) << -1, -1, -1);

        t = false;
        c = isPointOnLine_2D(Vec3f(v), Vec3f(l));
        if (t != c){
            cout << "isPointOnLine_2D - negative dot product: fail" << "\nExpected:\n"<< t << "\nGiven:\n" << c<<endl; 
            cout << "Press enter to continue..." << endl;
            cin.get();
            exit(-1);
        }
        
        v = (Mat_<float>(3,1) << 0, 1e-6, 1);
        l = (Mat_<float>(3,1) << -1, -1, 0);

        t = true;
        c = isPointOnLine_2D(Vec3f(v), Vec3f(l));
        if (t != c){
            cout << "isPointOnLine_2D - eps: fail" << "\nExpected:\n"<< t << "\nGiven:\n" << c<<endl; 
            cout << "Press enter to continue..." << endl;
            cin.get();
            exit(-1);
        }
        
        v = (Mat_<float>(3,1) << 5, -2, 0);
        l = (Mat_<float>(3,1) << 0, 0, 1);

        t = true;
        c = isPointOnLine_2D(Vec3f(v), Vec3f(l));
        if (t != c){
            cout << "isPointOnLine_2D - ideal line: fail" << "\nExpected:\n"<< t << "\nGiven:\n" << c<<endl; 
            cout << "Press enter to continue..." << endl;
            cin.get();
            exit(-1);
        }
            
    }catch(const std::exception &exc){
            cout <<  exc.what();
    }
}


void test_eucl2hom()
{
    cv::Vec2f a(1, 2);
    cv::Vec3f b = eucl2hom_point_2D(a);
    if ((a(0) * b(2) != b(0)) || 
        (a(1) * b(2) != b(1))) {
        cout << "There seems to be a problem with eucl2hom_point_2D(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }
}

void test_hom2eucl()
{
    cv::Vec3f a(8, 4, 2);
    cv::Vec2f b = hom2eucl_point_2D(a);
    if ((b(0) != 4.0f) || 
        (b(1) != 2.0f)) {
        cout << "There seems to be a problem with hom2eucl_point_2D(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }
}


int main(int argc, char** argv) {
    
    test_getConnectingLine();
    test_getScaleMatrix();
    test_getRotMatrix();
    test_getTranslMatrix();
    test_getH();
    test_applyH();
    test_isPointOnLine();
    test_eucl2hom();
    test_hom2eucl();
    
    cout << "Finished basic testing: Everything seems to be fine." << endl;
    cin.get();

    return 0;

}


