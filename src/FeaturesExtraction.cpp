/** 
    Definition of a class that extracts features from the images, to be used in the Detection step.
    @file FeaturesExtraction.cpp
    @author Alessandra Tonin
*/
#include <include/FeaturesExtraction.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;

/**
    Constructor.
*/
FeaturesExtraction::FeaturesExtraction()
{
}

/**
 * Extract HOG features from the given set of images.
 * @param wsize The size of the detection sliding window.
 * @param img_lst The set of images in which extract HOG.
 * @param gradient_lst The list of extracted gradients.
*/
void FeaturesExtraction::extractHOG(const Size wsize, const vector<Mat> &img_lst, vector<Mat> &gradient_lst)
{
    HOGDescriptor hog;
    hog.winSize = wsize;
    Mat gray;
    vector<float> descriptors;
    for (size_t i = 0; i < img_lst.size(); i++)
    {
        if (img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height)
        {
            Rect r = Rect((img_lst[i].cols - wsize.width) / 2,
                          (img_lst[i].rows - wsize.height) / 2,
                          wsize.width,
                          wsize.height);
            cvtColor(img_lst[i](r), gray, COLOR_BGR2GRAY);
            hog.compute(gray, descriptors, Size(8, 8), Size(0, 0));
            gradient_lst.push_back(Mat(descriptors).clone());
        }
    }
};
/**
 * Function to convert the extracted features in order to be used as training data for the models.
 * @param train_samples 
 * @param trainData
*/
void FeaturesExtraction::convert_to_ml(const vector<Mat> &train_samples, Mat &trainData)
{
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
    Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = Mat(rows, cols, CV_32FC1);
    for (size_t i = 0; i < train_samples.size(); ++i)
    {
        CV_Assert(train_samples[i].cols == 1 || train_samples[i].rows == 1);
        if (train_samples[i].cols == 1)
        {
            transpose(train_samples[i], tmp);
            tmp.copyTo(trainData.row((int)i));
        }
        else if (train_samples[i].rows == 1)
        {
            train_samples[i].copyTo(trainData.row((int)i));
        }
    }
};
