/** 
    Declaration of a class that extracts feature from the images, to be used in the classification step.
    @file FeaturesExtraction.hpp
    @author Alessandra Tonin
*/

using namespace cv;
using namespace std;

class FeaturesExtraction
{
    // Methods

public:
    // Constructor.
    FeaturesExtraction();

    // Function to extract HOG features of a set of images.
    void extractHOG(const Size wsize, const vector<Mat> &img_lst, vector<Mat> &gradient_lst);

    // Function to convert data to be used as training set.
    void convert_to_ml(const vector<Mat> &train_samples, Mat &trainData);
};