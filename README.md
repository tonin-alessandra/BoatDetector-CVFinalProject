# BoatDetector-CVFinalProject

This code implements a boat detector system based on extraction of Histogram of Oriented Gradients features and Support Vector Machines classifier.

Compiling the code may require some parameters, to be inserted through the command line.
When executing the project, you have two options:
          - simply test the trained detector, using the default images and ground-truth files provided together with the code
          - execute the complete code, retraining from scratch the model and then automatically testing it.
        
If you choose the first options (recommended), there's no need to provide any parameter, so simply compile as always and all the needed variables will get their 
default values.
If you decide to retrain the model, the parameters you should provide in order to make the code run correctly are:
 - pd   --> path of directory of positive images used to train the model,
 - nd   --> path of directory of negative images used to train the model,
 - td   --> path of directory of test images,
 - t    --> test a trained detector (default value is true).
pd and nd are not necessary if you use the already trained model, otherwise they will be the paths to you positive and negative samples used for the trining.
Also td is set to a default value, that is the relative path of the testSet directory of this project, containing the images used for training. If you want to change this
parameter, please be sure to provide the test images and the corresponding ground-truth using the same structure adopted in this project.
The structure required for the test set is: 
 - put your images into the project directory /testSet/images
 - put the ground-truth files into the project directory /testSet/ground_truth
 - images files must be named with a 2 digits progressive number and with jpg or png extension, like 00.jpg, 01.jpg, 02.jpg, ..., 10.jpg and so on
 - there must be one ground truth file for each image, named with the name of the corresponding image and .txt extension (for example, the ground truth file of image 00.jpg must
   be named 00.txt)
 - the .txt file must use the standard describe in the report provided with this project.

In order to avoid problems with the test of the detector, I suggest to choose the first option and just try the provided model of the detector, without re-training it. This suggestion is for two main reasons: the time required to train the model is not neglettable, and also, varying the training sets may lead to completely different results with respect to the current ones.
