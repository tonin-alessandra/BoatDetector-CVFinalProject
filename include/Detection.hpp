using namespace cv;
using namespace std;
using namespace ml;

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

class Detection
{
    // Data

protected:
    Ptr<SVM> svmModel;
    vector<float> svmVec;
    vector<vector<Rect>> totFoundRects;
    vector<vector<double>> totConfScores;

    // Methods

public:
    // Constructor.
    Detection();

    //Create and train svm
    Ptr<SVM> createSVM(Mat features, vector<int> labels);

    
    //Load and test image
    //void testImage(String svm, vector<Mat> testImgs, String result);

    //Get found rectangles
    vector<vector<Rect>> getRects();
    
    //
    vector<vector<double>> getConfidenceScores();
    
    //Get svm (example code)
    vector<float> get_svm_detector(const Ptr<SVM> &svm);

    //Test the trained detector 
    void testTrainedDetector(String obj_det_filename, vector<Mat> test_dir, String res_name);

private:
    //Load trained svm model
    void loadSVM(String path);
    // Draw rectangles to identify detected objects
    void drawRect(Mat img, vector<Rect> rects);

    
};