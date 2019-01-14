#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<string>
#include<iostream>
using namespace cv;

int main(int argc,char **argv){
    std::vector<cv::String> filenames;
    cv::glob(argv[1],filenames);
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);
    for(int i=0;i<filenames.size();i++){
        Mat im = imread(filenames[i]);
        cvtColor(im,im,COLOR_BGR2GRAY);

        equalizeHist(im,im);
        resize(im,im,Size(227,227));
        bool succ = imwrite(filenames[i],im,compression_params);
        std::cout<<filenames[i]<<std::endl;
        if(succ){
            // imshow("Hist eq",im);
            // waitKey(0);
        }
    }
}