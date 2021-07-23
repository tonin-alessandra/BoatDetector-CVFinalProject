/** 
    This class performs all utils operations.
    @file Utils.cpp
    @author Alessandra Tonin
*/
#include <include/Utils.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * Constructor
 */
Utils::Utils(){};

/**
 * Parse txt file of ground truth
 */ 
vector<int> Utils::parseTxtGT(String pathOfTxt){
    ifstream file_variable;
    file_variable.open(pathOfTxt);

    string line;
    vector<int> groundTruth;
    int rows = 0;
    string boat = "boat:";
    if (file_variable.is_open())
    {
        while (getline(file_variable, line))
        {

            istringstream stream(line);

            char separator = ';';
            string value;
           
            while (getline(stream, value, ';'))
            {
                if (value.find(boat) != string::npos)
                    value.erase(value.find(boat), boat.length());
                groundTruth.push_back(stoi(value));
            }
        }
        rows++;
        file_variable.close();
    }
    return groundTruth;
}