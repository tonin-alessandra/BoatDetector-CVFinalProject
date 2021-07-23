/** 
    This class performs the detection of objects based on the previously extracted features.
    @file Detection.cpp
    @author Alessandra Tonin
*/
#include <include/Detection.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;
using namespace ml;

/**
	Constructor.
*/
Detection::Detection(){};

/**
 * Create svm model and train it.
 * @param data Data used to construct the classifier.
 * @param label labels used to construct the classifier.
 * @return A pointer to the created SVM model.
*/
Ptr<SVM> Detection::createSVM(Mat data, vector<int> label)
{
    Ptr<SVM> svm = SVM::create();
    // Default values to train SVM.
    svm->setType(SVM::EPS_SVR);
    svm->setKernel(SVM::LINEAR);
    svm->setCoef0(0.0);
    svm->setDegree(3);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3));
    svm->setGamma(0);
    svm->setNu(0.5);
    svm->setP(0.1);
    svm->setC(0.01);
    svm->train(data, ROW_SAMPLE, label);
    return svm;
};
/**
 * Gets an HOG based detector.
 * @param svm The model of the SVM classifier used for detection.
 * @return The trained detector. 
*/
vector<float> Detection::get_svm_detector(const Ptr<SVM> &svm)
{
    Mat suppVecs = svm->getSupportVectors();
    const int sv_total = suppVecs.rows;
    Mat alpha, suppVecIdx;
    double rho = svm->getDecisionFunction(0, alpha, suppVecIdx);
    CV_Assert(alpha.total() == 1 && suppVecIdx.total() == 1 && sv_total == 1);
    CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
              (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
    CV_Assert(suppVecs.type() == CV_32F);
    vector<float> hog_detector(suppVecs.cols + 1);
    memcpy(&hog_detector[0], suppVecs.ptr(), suppVecs.cols * sizeof(hog_detector[0]));
    hog_detector[suppVecs.cols] = (float)-rho;
    return hog_detector;
}
/**
 * Test the trained detector on a set of new images.
 * @param obj_det_filename The name of the detector to test.
 * @param testImgs The images used as test set.
*/
void Detection::testTrainedDetector(String obj_det_filename, vector<Mat> testImgs)
{
    cout << "Testing trained detector..." << endl;
    HOGDescriptor hog;
    hog.load(obj_det_filename);
    obj_det_filename = "testing " + obj_det_filename;
    vector<vector<Rect>> totalDetections;
    vector<vector<double>> totalScores;
    for (int i = 0; i < testImgs.size(); i++)
    {
        Mat img;
        if (i < testImgs.size())
        {
            img = testImgs[i];
        }
        if (img.empty())
        {
            return;
        }
        vector<Rect> detections;
        vector<double> foundWeights;
        hog.detectMultiScale(img, detections, foundWeights, 0.5, Size(3, 3));
        totalDetections.push_back(detections);
        totalScores.push_back(foundWeights);
    }
    totFoundRects = totalDetections;
    totConfScores = totalScores;
}
/**
 * Retrieves the predicted rectangles.
 * @returns The predicted rectangles for the images of the test set.
 */
vector<vector<Rect>> Detection::getRects()
{
    return totFoundRects;
}

/**
 * Retrieves the confidence scores of the predicted rectangles.
 * @returns The confidence scores of the predicted rectangles for the images of the test set.
 */
vector<vector<double>> Detection::getConfidenceScores()
{
    return totConfScores;
}
