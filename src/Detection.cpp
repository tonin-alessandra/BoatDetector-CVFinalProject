// svm classifier, training and prediction, detection
/** 
    This class performs the  detection of objects based on the previously extracted features.
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
	Constructor
*/
Detection::Detection(){};

/**
	Create svm and train it
*/
/*void Detection::createSVM(Mat feature, Mat label)
{
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	//svm->setType(SVM::ONE_CLASS);
	svm->setKernel(SVM::LINEAR);
	Ptr<TrainData> tData = TrainData::create(feature, ROW_SAMPLE, label);
	//Ptr<TrainData> tData = TrainData::create(feature, COL_SAMPLE, label); ???????
	cout << "train begin" << endl;
	svm->trainAuto(tData);
	svm->save("C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/Tonin_FinalProject/SVM_Model.xml");
	cout << "train done" << endl;
	svmModel = svm;
};*/

/**
 * Load a trained svm model and its important parameters
*/
/*void Detection::loadSVM(String svmPath)
{
	Ptr<SVM> svm = SVM::load(svmPath);
	if (svm->empty())
	{
		cout << "Failed to read XML file." << endl;
	}
	else
	{
		cout << "Successfully read the XML file." << endl;
	}
	Mat suppVecs = svm->getSupportVectors(); //each vector is a float (row of a matrix) --> number of sv is suppVecs.rows
	int svmDim = svm->getVarCount();		 //number of variables in training samples
	Mat alpha = Mat::zeros(suppVecs.rows, svmDim, CV_32F); //weights of svm
	Mat svindex = Mat::zeros(1, suppVecs.rows, CV_64F); //indices of support vectors
	Mat result;
	double rho = svm->getDecisionFunction(0, alpha, svindex);
	alpha.convertTo(alpha, CV_32F);
	result = -1 * alpha * suppVecs;
	vector<float> vec;
	for (int i = 0; i < svmDim; ++i)
	{
		vec.push_back(result.at<float>(0, i));
	}
	vec.push_back(rho);
	ofstream fout("HOGDetector.txt");
	for (int i = 0; i < vec.size(); ++i)
	{
		fout << vec[i] << endl;
	}
	cout << "Save file detected by HOG complete" << endl;

	//return svm;
	svmVec = vec;
};*/


//--------------------------------------------------------------------------------
/**
	Create svm and train it
*/
Ptr<SVM> Detection::createSVM(Mat data, vector<int> label)
{
	Ptr<SVM> svm = SVM::create();
    /* Default values to train SVM */
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
write comment
*/
vector<float> Detection::get_svm_detector(const Ptr<SVM> &svm)
{
    // get the support vectors
    Mat suppVecs = svm->getSupportVectors();
    const int sv_total = suppVecs.rows;
    // get the decision function
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
Write comment
*/
void Detection::testTrainedDetector(String obj_det_filename, vector<Mat> testImgs, String resultName)
{
    cout << "Testing trained detector..." << endl;
    HOGDescriptor hog;
    hog.load(obj_det_filename);
    //vector<String> files;
    //glob(test_dir, files);

    obj_det_filename = "testing " + obj_det_filename;
    vector<vector<Rect>> totalDetections;
    for (int i = 0; i<testImgs.size(); i++)
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
        for (size_t j = 0; j < detections.size(); j++)
        {
            Scalar color = Scalar(0, foundWeights[j] * foundWeights[j] * 200, 0);
            rectangle(img, detections[j], color, img.cols / 400 + 1);
        }
        //imwrite("C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/Tonin_FinalProject/results/"+ resultName + to_string(i) + ".jpg", img);
    }
    totFoundRects = totalDetections;
}
/**
 * 
 */
vector<vector<Rect>> Detection::getRects(){
    return totFoundRects;
}
