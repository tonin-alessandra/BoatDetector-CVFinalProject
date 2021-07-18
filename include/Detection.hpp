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
    vector<Rect> foundRects;

    // Methods

public:
    // Constructor.
    Detection();

    //Create and train svm
    void createSVM(Mat feature, Mat label);

    
    //Load and test image
    void testImage(String svm, vector<Mat> testImgs, String result);

    //Get found rectangles
    vector<Rect> getRects();

private:
    //Load trained svm model
    void loadSVM(String path);
    // Draw rectangles to identify detected objects
    void drawRect(Mat img, vector<Rect> rects);
};