//============================================================================
// Name        : test.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================

#include <iostream>

#include "Pcv1.h"

using namespace std;

// usage: path to image in argv[1]
// main function. loads and saves image
int main(int argc, char** argv) {

    // will contain path to the input image (taken from argv[1])
    string fname;

    // check if image path was defined
    if (argc != 2){
        cout << "Usage: pcv1 <path_to_image>" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        return -1;
    }else{
        // if yes, assign it to variable fname
        fname = argv[1];
    }
    
    // start processing
    pcv1::run(fname);

    return 0;

}
