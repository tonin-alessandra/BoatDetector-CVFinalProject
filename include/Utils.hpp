/** 
    Declaration of a class that provide some utility methods.
    @file Utils.hpp
    @author Alessandra Tonin
*/

using namespace cv;
using namespace std;

class Utils
{
    // Data

protected:
    
    // Methods

public:
    // Constructor.
    Utils();
    
    //Parse a ground-truth txt file
    vector<int> parseTxtGT(String filePath);
};