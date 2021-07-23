/** 
    Declaration of a class that provides some utility methods.
    @file Utils.hpp
    @author Alessandra Tonin
*/

using namespace cv;
using namespace std;

class Utils
{
    // Methods

public:
    // Constructor.
    Utils();

    // Parse a ground-truth txt file.
    vector<int> parseTxtGT(String filePath, String folderPath);
};