// non maxima suppression
// elimination of false positive (+ add them to negative set)
// bounding box refinition
// performance evaluation iou (or separate class??)
/** 
    This class performs all the postprocessing operations needed after the detection part.
    @file Postprocessing.cpp
    @author Alessandra Tonin
*/
#include <include/Postprocessing.hpp>
#include <opencv2/opencv.hpp>
#include <assert.h>

using namespace cv;
using namespace std;

/**
 * Constructor.
 */
Postprocessing::Postprocessing(){};

/**
 * Compute non-maxima suppression to remove overlapping bounding boxes.
 */
void Postprocessing::nonMaxSuppression(const vector<Rect> &srcRects, const vector<double> &scores, vector<Rect> &resRects, float thresh, int neighbors = 0, double minScoresSum = 0)
{

    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
        return;

    assert(srcRects.size() == scores.size());

    // Sort the bounding boxes by the detection score
    multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.emplace(scores[i], i);
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --end(idxs);
        const Rect &rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;
        float scoresSum = lastElem->first;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs);)
        {
            // grab the current rectangle
            const cv::Rect &rect2 = srcRects[pos->second];

            float overlap = computeIOU(rect1, rect2);

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                scoresSum += pos->first;
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors && scoresSum >= minScoresSum)
            resRects.push_back(rect1);
    }
};

/**
 * Test the performance of the detector by using IoU metric.
 */
vector<float> Postprocessing::testPerformance(vector<Rect> groundTruth, vector<Rect> detections)
{
    // max obtained iou for each detected box
    vector<float> iouScores;
    for (int i = 0; i < detections.size(); i++)
    {
        vector<float> tmp;
        for (int j = 0; j < groundTruth.size(); j++)
        {
            float iou = computeIOU(groundTruth[i], detections[j]);
            tmp.push_back(iou);
        }
        float maxElem = tmp[0];
        for (int i = 0; i < tmp.size(); i++)
        {
            if (tmp[i] > maxElem)
                maxElem = tmp[i];
        }
        iouScores.push_back(maxElem);
    }
    return iouScores;
};

/**
 * Compute IoU between two bounding boxes.
 * @param box1 One of the two boxes. 
 * @param box2 One of the two boxes.
 */
float Postprocessing::computeIOU(Rect box1, Rect box2)
{
    float area1 = box1.area();
    float area2 = box2.area();
    float intersectionArea, unionArea;
    intersectionArea = (box1 & box2).area();
    unionArea = (area1 + area2) - intersectionArea;
    float iou = intersectionArea / unionArea;
    return iou;
};
