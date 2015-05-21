#ifndef SVM_DETECTOR_H
#define SVM_DETECTOR_H
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

class SVMDetector
{
	struct svm_model* model;
	vector<float> value_max, value_min;
	
	bool is_predict_probability;

	int SVMPredict(vector<float> & descript_vector, double *prob_est = NULL);

public:
	bool isTest;

	SVMDetector();

	void initDetector(string model_file_name);
	void detectInRectUsingOpenCVHog(const Mat & img, vector<Rect> & outputRects, vector<double> &scores, double zoom_scale = 1.2, int zoom_times = 3);
	
	~SVMDetector();
};

#endif