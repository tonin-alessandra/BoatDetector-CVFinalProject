using namespace cv;
using namespace std;
using namespace ml;

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

class Detection
{
    // Data

protected:
    vector<vector<Rect>> totFoundRects;
    vector<vector<double>> totConfScores;

    // Methods

public:
    // Constructor.
    Detection();

    //Create and train svm
    Ptr<SVM> createSVM(Mat features, vector<int> labels);

    //Get found rectangles
    vector<vector<Rect>> getRects();
    
    //Get consifence scores of predicted bounding boxes
    vector<vector<double>> getConfidenceScores();
    
    //Get svm (example code)
    vector<float> get_svm_detector(const Ptr<SVM> &svm);

    //Test the trained detector 
    void testTrainedDetector(String obj_det_filename, vector<Mat> test_dir, String res_name);
};