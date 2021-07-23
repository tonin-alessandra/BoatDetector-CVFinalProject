#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Postprocessing
{
    //Data
protected:
    // Methods

public:
    // Constructor.
    Postprocessing();

    //Perform non-maxima suppression
    void nonMaxSuppression(const vector<Rect>& srcRects, vector<Rect>& resRects, float thresh, int neighbors);

    //Compute Intersection over Union between boxes (IoU)
    float computeIOU(Rect b1, Rect b2);

    //Test performance of the detector: compute iou of the detected boxes wrt ground truth ones
    vector<float> testPerformance(vector<Rect> groundtruth, vector<Rect> detected);
};