/** 
    This is a boat detector based on classical computer vision techniques.
    @file main.cpp
    @author Alessandra Tonin
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <src/Preprocessing.cpp>
//#include <Preprocessing.hpp>

#include <src/Postprocessing.cpp>
//#include <Postprocessing.hpp>

#include <src/Detection.cpp>
//#include <include/Detection.hpp>

#include <src/FeaturesExtraction.cpp>
//#include <include/FeaturesExtraction.hpp>


using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    //String positivePath = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/FINAL_DATASET/FINAL_DATASET/TRAINING_DATASET/IMAGES";
    //String negativePath = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/named_neg";
    String positivePath = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/boat";
    String negativePath = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/not_boat";
    String svmModelPath = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/Tonin_FinalProject/SVM_Model.xml";
    String veniceTestPath = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/FINAL_DATASET/FINAL_DATASET/TEST_DATASET/venice";
    String kaggleTestPath = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/FINAL_DATASET/FINAL_DATASET/TEST_DATASET/kaggle";
    String resultsVenicePath = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/Tonin_FinalProject/results/venice";
    String resultsKagglePath = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/Tonin_FinalProject/results/kaggle";
    vector<Mat> positiveImgs;
    vector<Mat> negativeImgs;
    vector<Mat> veniceTestImgs;
    vector<Mat> kaggleTestImgs;
    vector<Mat> equalized_pos;

    // Instantiate an object of Preprocessing class.
    Preprocessing preprocessor = Preprocessing();

    // load images
    preprocessor.loadImgs(positivePath, positiveImgs);
    preprocessor.loadImgs(negativePath, negativeImgs);
    preprocessor.loadImgs(veniceTestPath, veniceTestImgs);
    preprocessor.loadImgs(kaggleTestPath, kaggleTestImgs);

    // enhance contrast
    /*preprocessor.equalizeImgs(positiveImgs, equalized_pos);
    cout<<"a";
    imshow("original", positiveImgs[25]);
    cout<<"b";
    waitKey();
    cout<<"w";
    imshow("equalized", equalized_pos[25]);
    cout<<"c";
    waitKey();*/

    // extract HOG features from both positive and negative images
    FeaturesExtraction featExtractor = FeaturesExtraction();
    featExtractor.extractHOG(positiveImgs, negativeImgs);

    // creat and train the svm classifier
    Detection svmClass = Detection();
    svmClass.createSVM(featExtractor.getFeature(), featExtractor.getLabel());

    // test the detector in both datasets' images
    svmClass.testImage(svmModelPath, kaggleTestImgs, resultsKagglePath);
    svmClass.testImage(svmModelPath, veniceTestImgs, resultsVenicePath);
}
