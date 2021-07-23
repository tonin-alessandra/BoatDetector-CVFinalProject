/** 
    This is a boat detector program based on classical computer vision techniques.
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
#include <src/Utils.cpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

/**
 * Main method.
 */
int main(int argc, char **argv)
{
    // Parse the input given on the Command Line.
    const char *keys =
        {
            "{pd    |     | path of directory contains positive images}"
            "{nd    |     | path of directory contains negative images}"
            "{td    |     | path of directory contains test images}"
            "{t     |true| test a trained detector}"};
    CommandLineParser parser(argc, argv, keys);

    String obj_det_filename = "HOGboats.xml";
    bool train_twice = true;

    Preprocessing preprocessor = Preprocessing();
    FeaturesExtraction hogExtractor = FeaturesExtraction();
    Detection detector = Detection();
    Postprocessing postprocessor = Postprocessing();
    Utils utilities = Utils();

    Size newSize = Size(130, 90);
    vector<Mat> pos_lst, full_neg_lst, neg_lst, gradient_lst;
    vector<int> labels;
    Mat train_data;
    HOGDescriptor hog;
    String groundTruthPath = "testSet/ground_truth/";

    // Parsing of cmd line.
    String pos_dir = parser.get<String>("pd");
    String neg_dir = parser.get<String>("nd");
    String test_dir = parser.get<String>("td");
    bool test_detector = parser.get<bool>("t");

    if ((pos_dir.empty() || neg_dir.empty()) && !(test_detector))
    {
        // These lines are an example of paths in case of problems with the command line inputs.
        //pos_dir = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/boat";
        //neg_dir = "C:/Users/ASUS/Documents/magistrale/first_year/computer_vision/final_project/named_neg";
    }
    if (test_dir.empty())
    {
        test_dir = "testSet/images";
    }
    if (!test_detector)
    {
        test_detector = true;
    }
    if (test_detector)
    {
        vector<Mat> test;
        preprocessor.loadImages(test_dir, test);

        // ***********************************for each test image, load the ground truth from txt file
        vector<vector<Rect>> totGT;

        for (int i = 0; i < test.size(); i++)
        {
            vector<Rect> currentGtRects;
            String filename;
            if (i < 10)
                filename = "0" + to_string(i) + ".txt";
            else
                filename = to_string(i) + ".txt";

            vector<int> gtCoords = utilities.parseTxtGT(filename, groundTruthPath);

            for (int j = 0; j < gtCoords.size(); j += 4)
            {
                currentGtRects.push_back(Rect(gtCoords[j], gtCoords[j + 2], gtCoords[j + 1] - gtCoords[j], gtCoords[j + 3] - gtCoords[j + 2]));
            }
            totGT.push_back(currentGtRects);
        }
        //**********************************************end of gt loading
        detector.testTrainedDetector(obj_det_filename, test);
        vector<vector<Rect>> detectedRect = detector.getRects();
        vector<vector<double>> detectedScores = detector.getConfidenceScores();
        // *************************perform nms
        vector<vector<Rect>> nmsResRects;
        for (int i = 0; i < detectedRect.size(); i++)
        {
            vector<Rect> tmp;
            postprocessor.nonMaxSuppression(detectedRect[i], detectedScores[i], tmp, 0.03, 0, 0);
            nmsResRects.push_back(tmp);
        }
        //******************end of nms

        //***************************evaluate iou
        vector<vector<float>> iouScores;
        for (int g = 0; g < nmsResRects.size(); g++)
        {
            vector<float> scores = postprocessor.testPerformance(totGT[g], nmsResRects[g]);
            iouScores.push_back(scores);
        }
        //*************************end of iou evaluation

        //**************** draw detected bb and gt bb

        for (int i = 0; i < test.size(); i++)
        {
            Mat image = test[i];
            for (int j = 0; j < nmsResRects[i].size(); j++)
            {
                // Draw in green all the detected rectangles.
                rectangle(image, nmsResRects[i][j], Scalar(0, 255, 0), 5);
                putText(image, to_string(iouScores[i][j]), nmsResRects[i][j].tl(), HersheyFonts::FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            }

            for (int u = 0; u < totGT[i].size(); u++)
            {
                // Draw in blue all the ground-truth rectangles.
                rectangle(image, totGT[i][u], Scalar(255, 0, 0), 5);
            }
            imwrite("results/" + to_string(i) + ".jpg", image);
            imshow("Image", image);
            waitKey();
        }
        //**************** end of drawing
        exit(0);
    }

    // The code from this point is needed just for the creation and training of the model.

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

    // Compute Histogram of Oriented Gradients for both positive and negative images.
    clog << "Computing HoG for positive images...";
    hogExtractor.extractHOG(pos_image_size, pos_lst, gradient_lst);
    size_t positive_count = gradient_lst.size();
    labels.assign(positive_count, +1);
    clog << "...[done] ( positive images count : " << positive_count << " )" << endl;
    clog << "Computing HoG for negative images...";
    hogExtractor.extractHOG(pos_image_size, full_neg_lst, gradient_lst);
    size_t negative_count = gradient_lst.size() - positive_count;
    labels.insert(labels.end(), negative_count, -1);
    CV_Assert(positive_count < labels.size());
    clog << "...[done] ( negative images count : " << negative_count << " )" << endl;

    // Prepare data for the training phase.
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
    vector<Mat> test;
    preprocessor.loadImages(test_dir, test);

    // For each test image, load the ground truth from txt file.
    vector<vector<Rect>> totGT;
    for (int i = 0; i < test.size(); i++)
    {
        vector<Rect> currentGtRects;
        String filename;
        if (i < 10)
            filename = "0" + to_string(i) + ".txt";
        else
            filename = to_string(i) + ".txt";
        vector<int> gtCoords = utilities.parseTxtGT(filename, groundTruthPath);
        for (int j = 0; j < gtCoords.size(); j += 4)
        {
            currentGtRects.push_back(Rect(gtCoords[j], gtCoords[j + 2], gtCoords[j + 1] - gtCoords[j], gtCoords[j + 3] - gtCoords[j + 2]));
        }
        totGT.push_back(currentGtRects);
    }

    detector.testTrainedDetector(obj_det_filename, test);
    vector<vector<Rect>> detectedRect = detector.getRects();
    vector<vector<double>> detectedScores = detector.getConfidenceScores();

    // Perform Non-Maxima Suppression.
    vector<vector<Rect>> nmsResRects;
    for (int i = 0; i < detectedRect.size(); i++)
    {
        vector<Rect> tmp;
        postprocessor.nonMaxSuppression(detectedRect[i], detectedScores[i], tmp, 0.03, 0, 0);
        nmsResRects.push_back(tmp);
    }

    // Evaluate iou.
    vector<vector<float>> iouScores;
    for (int g = 0; g < nmsResRects.size(); g++)
    {
        vector<float> scores = postprocessor.testPerformance(totGT[g], nmsResRects[g]);
        iouScores.push_back(scores);
    }

    // Draw detected bounding boxes and true ones.

    for (int i = 0; i < test.size(); i++)
    {
        Mat image = test[i];
        for (int j = 0; j < nmsResRects[i].size(); j++)
        {
            // Draw in green all the detected rectangles.
            rectangle(image, nmsResRects[i][j], Scalar(0, 255, 0), 5);
            putText(image, to_string(iouScores[i][j]), nmsResRects[i][j].tl(), HersheyFonts::FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        }

        for (int u = 0; u < totGT[i].size(); u++)
        {
            // Draw in blue all the ground-truth rectangles.
            rectangle(image, totGT[i][u], Scalar(255, 0, 0), 5);
        }
        imwrite("results/" + to_string(i) + ".jpg", image);
        imshow("Image", image);
        waitKey();
    }
    return (0);
}