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
    Mat img;
    for (int i = 0; i < imgs.size(); i++)
    {
        img = imgs[i];
        vector<Mat> rgb_channels;
        split(img, rgb_channels);
        Mat eq_r_channel, eq_g_channel, eq_b_channel;
        equalizeHist(rgb_channels[0], eq_r_channel);
        equalizeHist(rgb_channels[1], eq_g_channel);
        equalizeHist(rgb_channels[2], eq_b_channel);
        vector<Mat> equal_channels;
        equal_channels.push_back(eq_r_channel);
        equal_channels.push_back(eq_g_channel);
        equal_channels.push_back(eq_b_channel);
        Mat equal_image;
        merge(equal_channels, equal_image);
        /*cvtColor(img, img, COLOR_BGR2GRAY);
        Mat img2;
        equalizeHist(img, img2);
        cvtColor(img2, img2, COLOR_GRAY2BGR);*/
        eqImgs.push_back(equal_image);
    }
};

/**
    Function to remove noise
*/
void denoiseImgs(vector<Mat> imgs){};

/**
    Function to resize images
*/
void resizeImgs(vector<Mat> imgs){};

/**
    Function to rename images
*/
//void renameImgs(vector<Mat> images){};