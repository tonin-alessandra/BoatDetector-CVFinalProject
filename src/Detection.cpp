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
void Detection::createSVM(Mat feature, Mat label)
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
};

/**
 * Load a trained svm model and its important parameters
*/
void Detection::loadSVM(String svmPath)
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
};

/*
	Test the detector on an image
*/
void Detection::testImage(String svmPath, vector<Mat> testImages, String result)
{
	loadSVM(svmPath);
	HOGDescriptor hogTest(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	//HOGDescriptor hogTest;
	hogTest.setSVMDetector(svmVec);
	Mat img;
	vector<Rect> found, found_filtered;
	for (int i = 0; i < testImages.size(); i++)
	{
		img = testImages[i];
		if (!img.data)
		{
			cout << "Failed to read test picture" << endl;
		}

		int p = 2;
		resize(img, img, Size(img.cols / p, img.rows / p));
		hogTest.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		cout << endl
			 << "The size of the rectangular box is: " << found.size() << endl;
		drawRect(img, found);
		imwrite(result + "/res" + to_string(i) + ".jpg", img);
		//namedWindow("Img", 0);
		//imshow("Img"+to_string(i), img);
		//waitKey(0);
	}
	//foundRects = found;
};

/**
 * Draw the detected rectangles in the image
 */

void Detection::drawRect(Mat testImg, vector<Rect> rectFound)
{
	//vector<Rect> rects = getRects();
	for (int i = 0; i < rectFound.size(); i++)
	{
		Rect r = rectFound[i];

		r.x += cvRound(r.width * 0.1); //int cvRound(double value) Rounds a double type number and returns an integer number!
		r.width = cvRound(r.width * 0.8);
		r.y += cvRound(r.height * 0.07);
		r.height = cvRound(r.height * 0.8);

		rectangle(testImg, r.tl(), r.br(), Scalar(0, 0, 255), 3);
	}
};

/**
 * Returns the found rectangles
 */
vector<Rect> Detection::getRects()
{
	return foundRects;
};
