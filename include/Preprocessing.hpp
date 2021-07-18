/** 
    Declaration of a class that performs all the preprocessing operations needed before feature extraction and Detection parts.
    @file Preprocessing.hpp
    @author Alessandra Tonin
*/
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Preprocessing
{
    // Data

protected:

    // Methods

public:
    // Constructor.
    Preprocessing();

    //Function to load images
    void loadImgs(const String &dirname, vector<Mat> &images);

    // Function to equalize images.
    void equalizeImgs(vector<Mat> images, vector<Mat> dstImgs);

    //Function to remove noise
    void denoiseImgs(vector<Mat> images);

    //Function to resize images
    void resizeImgs(vector<Mat> images);

    //Function to rename images
    //void renameImgs(vector<Mat> images);
};