# BoatDetector-CVFinalProject

This code implements a boat detector system based on extraction of Histogram of Gradients features and Support Vector Machines classifier.

If the project is compiled from the terminal, a set of parameters must be provided in order to make the code run correctly.
In particular, the available commands are:
          pd   --> path of directory of positive images used to train the model
          nd   --> path of directory of negative images used to train the model
          td   --> path of directory of test images
          t    --> test a trained detector (default value is true)

Commands pd and nd are mandatory if you want to retrain the model, so if t is set to false. Otherwise, if t is set to true (to avoid retraining the model from scratch), you can 
omit also pd and nd, since they're not needed.
Also td is set to a default value, that is the relative path of the testSet directory of this project, containing the images used for training. If you want to change this
parameter, please be sure to provide the test images and the corresponding ground-truth using the same structure adopted in this project.
The structure required for the test set is: 
 - put your images into the project directory /testSet/images
 - put the ground-truth files into the project directory /testSet/ground_truth
 - images files must be named with a 2 digits progressive number and with jpg or png extension, like 00.jpg, 01.jpg, 02.jpg, ..., 10.jpg and so on
 - there must be one ground truth file for each image, named with the name of the corresponding image and .txt extension (for example, the ground truth file of image 00.jpg must
   be named 00.txt)
 - the .txt file must use the standard describe in the report file.
In order to avoid problems with the test of the detector, I suggest to use the default value for the test images path.
