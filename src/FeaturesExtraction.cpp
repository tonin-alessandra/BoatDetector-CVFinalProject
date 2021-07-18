// feature extraction
// sliding window
// image pyramid
/** 
    Definition of a class that extracts feature from the images, to be used in the Detection step.
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
    sample_feature;
    sample_label;
}

/**
 * Extract HOG features from the given set of images.
*/
void FeaturesExtraction::extractHOG(vector<Mat> posImages, vector<Mat> negImages)
{
    Mat sample_feature_mat;
    Mat sample_label_mat;
    int feature_dim;

    HOGDescriptor *hog = new HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    //HOGDescriptor *hog = new HOGDescriptor();

    //pos
    for (int i = 0; i < posImages.size(); ++i)
    {
        Mat train_data(64, 128, CV_32FC1);
        resize(posImages[i], train_data, Size(64, 128));
        vector<float> descriptor;
        hog->compute(train_data, descriptor, Size(8, 8));
        if (i == 0)
        {
            feature_dim = descriptor.size();
            sample_feature_mat = Mat::zeros(posImages.size() + negImages.size(), feature_dim, CV_32FC1);
            sample_label_mat = Mat::zeros(posImages.size() + negImages.size(), 1, CV_32SC1);
        }
        float *pf = sample_feature_mat.ptr<float>(i);
        int *pl = sample_label_mat.ptr<int>(i);
        for (int j = 0; j < feature_dim; ++j)
        {
            *pf++ = descriptor[j];
        }
        *pl++ = 1;
    }
    cout<<"positive done";
    //neg
    for (int i = 0; i < negImages.size(); ++i) {
		Mat train_data(64, 128, CV_32FC1);
		resize(negImages[i], train_data, Size(64, 128));
		vector<float>descriptor;
		hog->compute(train_data, descriptor, Size(8, 8));
		float *pf = sample_feature_mat.ptr<float>(i + posImages.size());
		int *pl = sample_label_mat.ptr<int>(i + posImages.size());
		for (int j = 0; j < feature_dim; ++j) {
			*pf++ = descriptor[j];
		}
		*pl++ = -1;
	}
    cout<<"negative done";

sample_feature = sample_feature_mat;
sample_label = sample_label_mat;
};

/**
 * //write comment
*/
Mat FeaturesExtraction::getFeature(){
    return sample_feature;
};

/**
 * //write comment
*/  
Mat FeaturesExtraction::getLabel(){
    return sample_label;
};;
