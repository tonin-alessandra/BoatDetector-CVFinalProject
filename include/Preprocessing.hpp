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
    // Methods

public:
    // Constructor.
    Preprocessing();

    // Function to load images.
    void loadImages(const String &dirname, vector<Mat> &images);

    // Function to remove noise.
    void denoiseImgs(vector<Mat> images, vector<Mat> denoised);

    // Function to resize images.
    vector<Mat> resizeImgs(vector<Mat> images, Size size);
};