/** 
    Declaration of a class that executes the postprocessing operations.
    @file Utils.hpp
    @author Alessandra Tonin
*/
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Postprocessing
{
    // Methods

public:
    // Constructor.
    Postprocessing();

    // Perform non-maxima suppression.
    void nonMaxSuppression(const vector<Rect> &srcRects, const vector<double> &scores, vector<Rect> &resRects, float thresh, int neighbors, double minScoreSum);

    // Compute Intersection over Union between boxes (IoU).
    float computeIOU(Rect b1, Rect b2);

    // Test performance of the detector: compute iou of the detected boxes wrt ground truth ones.
    vector<float> testPerformance(vector<Rect> groundtruth, vector<Rect> detected);
};