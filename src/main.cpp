/** 
    This is a boat detector based on classical computer vision techniques.
    @file main.cpp
    @author Alessandra Tonin
*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <src/Preprocessing.cpp>
#include <src/FeaturesExtraction.cpp>
#include <src/Detection.cpp>
#include <src/Postprocessing.cpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

/**
 * Main method
 */
int main(int argc, char **argv)
{
    // Parse the input given on the Command Line
    const char *keys =
        {
            "{help h|     | show help message}"
            "{pd    |     | path of directory contains positive images}"
            "{nd    |     | path of directory contains negative images}"
            "{td    |     | path of directory contains test images}"
            "{d     |false| train twice}"
            "{t     |false| test a trained detector}"
            "{fn    |my_detector.xml| file name of trained SVM}"};
    CommandLineParser parser(argc, argv, keys);

    String pos_dir = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/boat";
    String neg_dir = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/named_neg";
    String test_dirk = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/FINAL_DATASET/FINAL_DATASET/TEST_DATASET/kaggle";
    String test_dirv = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/FINAL_DATASET/FINAL_DATASET/TEST_DATASET/venice";
    String obj_det_filename = "HOGboats.xml";
    bool test_detector = true;
    bool train_twice = true;

    Preprocessing preprocessor = Preprocessing();
    FeaturesExtraction hogExtractor = FeaturesExtraction();
    Detection detector = Detection();
    Postprocessing postprocessor = Postprocessing();

    Size newSize = Size(130, 90);
    vector<Mat> pos_lst, full_neg_lst, neg_lst, gradient_lst;
    vector<int> labels;
    Mat train_data;
    HOGDescriptor hog;

    if (test_detector)
    {
        // need real gt parsed from xml/txt here
        vector<vector<Rect>> totGT;
        vector<Rect> gtRects;
        gtRects.push_back(Rect(607, 468, 950 - 607, 745 - 468));
        gtRects.push_back(Rect(800, 500, 950 - 800, 745 - 500));

        //fake gt, same 2 rects for all images
        for (int s = 0; s < 12; s++)
            totGT.push_back(gtRects);

        vector<Mat> test;
        preprocessor.loadImages(test_dirk, test);
        cout << "w";
        detector.testTrainedDetector(obj_det_filename, test, "zzz");
        cout << "y";
        vector<vector<Rect>> detectedRect = detector.getRects();
        cout << "z";
        vector<vector<Rect>> nmsResRects;
        cout << "e";
        for (int i = 0; i < detectedRect.size(); i++)
        {
            cout << "q";
            vector<Rect> tmp;
            postprocessor.nonMaxSuppression(detectedRect[i], tmp, 0.02, 0.0);
            nmsResRects.push_back(tmp);
        }
        vector<vector<float>> totScores;
        for (int g = 0; g <nmsResRects.size(); g++)
        {
            vector<float> scores = postprocessor.testPerformance(totGT[g], nmsResRects[g]);
            totScores.push_back(scores);
        }
        cout << "a";
        for (int i = 0; i < test.size(); i++)
        {
            Mat image = test[i];
            cout << "b";
            for (int j = 0; j < nmsResRects[i].size(); j++)
            {
                //red for detected
                rectangle(image, nmsResRects[i][j], Scalar(0, 0, 255), 10);
                putText(image, to_string(totScores[i][j]), nmsResRects[i][j].tl(), HersheyFonts::FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255),2 );
                cout << "c";
            }
            cout << "d";
            imwrite("C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/Tonin_FinalProject/results/vvv" + to_string(i) + ".jpg", image);
            cout << "e";
            imshow("img", image);
            waitKey();
        }
        /*for (int i = 0; i < gtRects.size(); i++)
        {
            //blue for ground truth
            rectangle(image, gtRects[i], Scalar(255, 0, 0), 10);
        }*/

        /*String filename = "image0018.xml";

        FileStorage sourceXMLFile;
        sourceXMLFile.open(filename, FileStorage::READ);
        FileNode n = sourceXMLFile.root();
        
        int type = n.type();
        cout<<to_string(type);
        bool str = n.empty();
        if (str)
            cout << "yes";
        else
        {
            cout << "No";
        }
        // Read mappings from a sequence

        FileNode n2 = sourceXMLFile["annotation"];
        //cout << "Width  " << n2["size"]["width"] << endl << endl;
        sourceXMLFile.release();*/
        exit(0);
    }

    if (pos_dir.empty() || neg_dir.empty())
    {
        parser.printMessage();
        cout << "Wrong number of parameters.\n\n"
             << "Example command line:\n"
             << argv[0] << " -dw=64 -dh=128 -pd=/INRIAPerson/96X160H96/Train/pos -nd=/INRIAPerson/neg -td=/INRIAPerson/Test/pos -fn=HOGpedestrian64x128.xml -d\n"
             << "\nExample command line for testing trained detector:\n"
             << argv[0] << " -t -fn=HOGpedestrian64x128.xml -td=/INRIAPerson/Test/pos";
        exit(1);
    }

    // Load and process positive images set.
    clog << "Positive images are being loaded...";
    preprocessor.loadImages(pos_dir, pos_lst);
    pos_lst = preprocessor.resizeImgs(pos_lst, newSize);
    preprocessor.denoiseImgs(pos_lst, pos_lst);

    // Do some checks on the loaded images.
    if (pos_lst.size() > 0)
    {
        clog << "...[done] " << pos_lst.size() << " files." << endl;
    }
    else
    {
        clog << "no image in " << pos_dir << endl;
        return 1;
    }
    Size pos_image_size = pos_lst[0].size();

    for (size_t i = 0; i < pos_lst.size(); ++i)
    {
        if (pos_lst[i].size() != pos_image_size)
        {
            cout << "All positive images should be same size!" << endl;
            exit(1);
        }
    }
    pos_image_size = pos_image_size / 8 * 8;

    // Load and process negative images set.
    clog << "Negative images are being loaded...";
    preprocessor.loadImages(neg_dir, full_neg_lst);
    full_neg_lst = preprocessor.resizeImgs(full_neg_lst, newSize);
    preprocessor.denoiseImgs(full_neg_lst, full_neg_lst);
    clog << "...[done] " << full_neg_lst.size() << " files." << endl;

    // Compute Histogram of Gradients for both positive and negative images.
    clog << "Computing HoG for positive images...";
    hogExtractor.extractHOG(pos_image_size, pos_lst, gradient_lst);
    size_t positive_count = gradient_lst.size();
    labels.assign(positive_count, +1);
    clog << "...[done] ( positive images count : " << positive_count << " )" << endl;
    clog << "Computing HoG for negative images...";
    hogExtractor.extractHOG(pos_image_size, full_neg_lst, gradient_lst);

    // write explanation.
    size_t negative_count = gradient_lst.size() - positive_count;
    labels.insert(labels.end(), negative_count, -1);
    CV_Assert(positive_count < labels.size());
    clog << "...[done] ( negative images count : " << negative_count << " )" << endl;

    // Prepara data for training phase.
    hogExtractor.convert_to_ml(gradient_lst, train_data);

    // Create and train the SVM model.
    clog << "Creating and training SVM...";
    Ptr<SVM> svm = detector.createSVM(train_data, labels);
    clog << "...[done]" << endl;

    if (train_twice)
    {
        clog << "Testing trained detector on negative images. This might take a few minutes...";
        HOGDescriptor my_hog;
        my_hog.winSize = pos_image_size;
        // Set the trained svm to my_hog
        my_hog.setSVMDetector(detector.get_svm_detector(svm));
        vector<Rect> detections;
        vector<double> foundWeights;
        for (size_t i = 0; i < full_neg_lst.size(); i++)
        {
            if (full_neg_lst[i].cols >= pos_image_size.width && full_neg_lst[i].rows >= pos_image_size.height)
                my_hog.detectMultiScale(full_neg_lst[i], detections, foundWeights, 0.5, Size(3, 3));
            else
                detections.clear();
            for (size_t j = 0; j < detections.size(); j++)
            {
                Mat detection = full_neg_lst[i](detections[j]).clone();
                resize(detection, detection, pos_image_size, 0, 0, INTER_LINEAR_EXACT);
                full_neg_lst.push_back(detection);
            }
        }
        clog << "...[done]" << endl;
        gradient_lst.clear();
        clog << "Histogram of Gradients are being calculated for positive images...";
        hogExtractor.extractHOG(pos_image_size, pos_lst, gradient_lst);
        positive_count = gradient_lst.size();
        clog << "...[done] ( positive count : " << positive_count << " )" << endl;
        clog << "Histogram of Gradients are being calculated for negative images...";
        hogExtractor.extractHOG(pos_image_size, full_neg_lst, gradient_lst);
        negative_count = gradient_lst.size() - positive_count;
        clog << "...[done] ( negative count : " << negative_count << " )" << endl;
        labels.clear();
        labels.assign(positive_count, +1);
        labels.insert(labels.end(), negative_count, -1);
        clog << "Training SVM again...";
        hogExtractor.convert_to_ml(gradient_lst, train_data);
        svm->train(train_data, ROW_SAMPLE, labels);
        clog << "...[done]" << endl;
    }

    // Test the trained HoG detector on the test set.
    hog.winSize = pos_image_size;
    hog.setSVMDetector(detector.get_svm_detector(svm));
    hog.save(obj_det_filename);
    String namek = "resk";
    String namev = "resv";
    //detector.testTrainedDetector(obj_det_filename, test_dirk, namek);
    //detector.testTrainedDetector(obj_det_filename, test_dirv, namev);
    vector<Mat> test;
    preprocessor.loadImages(test_dirk, test);
    cout << "w";
    detector.testTrainedDetector(obj_det_filename, test, "zzz");
    cout << "y";
    vector<vector<Rect>> detectedRect = detector.getRects();
    cout << "z";
    vector<vector<Rect>> nmsResRects;
    cout << "e";
    for (int i = 0; i < detectedRect.size(); i++)
    {
        cout << "q";
        vector<Rect> tmp;
        postprocessor.nonMaxSuppression(detectedRect[i], tmp, 0.02, 0.0);
        nmsResRects.push_back(tmp);
    }
    cout << "a";
    for (int i = 0; i < test.size(); i++)
    {
        Mat image = test[i];
        cout << "b";
        for (int j = 0; j < nmsResRects[i].size(); j++)
        {
            //red for detected
            rectangle(image, nmsResRects[i][j], Scalar(0, 0, 255), 10);
            cout << "c";
        }
        cout << "d";
        imwrite("C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/Tonin_FinalProject/results/kkk" + to_string(i) + ".jpg", image);
        cout << "e";
        imshow("img", image);
        waitKey();
    }
    return 0;
}