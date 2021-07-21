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
    @param dirname Path from where to load the images.
    @param img_lst Vector in which images will be saved.
*/
void Preprocessing::loadImages(const String &dirname, vector<Mat> &img_lst)
{
    vector<String> files;
    glob(dirname, files);
    for (size_t i = 0; i < files.size(); ++i)
    {
        Mat img = imread(files[i]);
        if (img.empty())
        {
            cout << files[i] << " is invalid!" << endl; 
            continue;
        }
        img_lst.push_back(img);
    }

};

/**
    Function to equalize images, to improve contrast.
    @param imgs The vector of images to equalize.
    @param eqImgs The vector of equalized images.
*/
/*void Preprocessing::equalizeImgs(vector<Mat> imgs, vector<Mat> eqImgs)
{
    for (int i = 0; i < imgs.size(); i++)
    {
        Mat img = imgs[i];
        cvtColor(img, img, COLOR_BGR2GRAY);
        equalizeHist(img, img);
        cvtColor(img, img, COLOR_GRAY2BGR);
        eqImgs.push_back(img);
    }
};*/

/**
    Function to remove noise.
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
    @param imgs The vector of images to resize.
    @param newSize The new size to give to images.
    @return A vector of resized images.

*/
vector<Mat> Preprocessing::resizeImgs(vector<Mat> imgs, Size newSize)
{
    vector<Mat> resized;
    for (int i = 0; i < imgs.size(); i++)
    {
        resize(imgs[i], imgs[i], newSize);
        resized.push_back(imgs[i]);
    }
    return resized;
};