/** 
    Declaration of a class that extracts feature from the images, to be used in the classificatino step.
    @file FeaturesExtraction.hpp
    @author Alessandra Tonin
*/

using namespace cv;
using namespace std;

class FeaturesExtraction
{
    // Data

protected:
    Mat sample_feature;
    Mat sample_label;

    // Methods

public:
    // Constructor.
    FeaturesExtraction();

    //Function to extract HOG features of a set of images.
    void extractHOG(vector<Mat> pos_image, vector<Mat> neg_imagess);

    Mat getFeature();

    Mat getLabel();
};