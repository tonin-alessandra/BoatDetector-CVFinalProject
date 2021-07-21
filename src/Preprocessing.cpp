/** 
    Definition of a class that performs all the preprocessing operations needed before feature extraction and Detection parts.
    @file Preprocessing.cpp
    @author Alessandra Tonin
*/
#include <include/Preprocessing.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

/**
    Constructor.
*/
Preprocessing::Preprocessing()
{
}

/**
    Load images from the specified directory.
    @param path Path from where to load the images.
    @param imgs Vector in which images will be saved.
*/
void Preprocessing::loadImgs(const String &path, vector<Mat> &imgs)
{
    vector<String> files;
    glob(path, files);
    for (int i = 0; i < files.size(); ++i)
    {
        Mat img = imread(files[i]); // load the image
        //if (!img.data) return -1;
        if (!img.data)
        {
            cout << files[i] << " is invalid!" << endl;
            continue;
        }
        imgs.push_back(img);
    }
};

/**
    Function to equalize images, to improve contrast.
    @param imgs The vector of images to equalize.
    @param eqImgs The vector of equalized images.
*/
void Preprocessing::equalizeImgs(vector<Mat> imgs, vector<Mat> eqImgs)
{
    for (int i = 0; i < imgs.size(); i++)
    {
        Mat img = imgs[i];
        cvtColor(img, img, COLOR_BGR2GRAY);
        equalizeHist(img, img);
        cvtColor(img, img, COLOR_GRAY2BGR);
        eqImgs.push_back(img);
    }
};

/**
    Function to remove noise
    @param imgs The vector of images to be denoised.
    @param denoisImgs The vector of denoised images.    
*/
void Preprocessing::denoiseImgs(vector<Mat> imgs, vector<Mat> denoisImgs)
{
    for (int i = 0; i < imgs.size(); i++)
    {
        //GaussianBlur(imgs[i], imgs[i], Size(0,0), 10);
        Mat img = imgs[i];
        fastNlMeansDenoisingColored(img, img);
        denoisImgs.push_back(img);
    }
};

/**
    Function to resize a set of images.
    @param imgs The vector of images to be resized.
    @param resizedImgs The vector of resized images.
    @param newSize The new size to give to images.
*/
void Preprocessing::resizeImgs(vector<Mat> imgs, vector<Mat> resizedImgs, Size newSize)
{
    for (int i = 0; i < imgs.size(); i++)
    {
        resize(imgs[i], imgs[i], newSize);
        resizedImgs.push_back(imgs[i]);
    }
};