#include <iostream>
#include <list>
#include <fstream>

#include "SVMDetector.h"
#include "svm.h"
#include "rectangles.h"

SVMDetector::SVMDetector()
{
	this->isTest = false;
	this->is_predict_probability = false;
}

void SVMDetector::initDetector(string model_file_name)
{
	model = svm_load_model(model_file_name.c_str());
	if (model == NULL)
	{
		cout << "can't open model file" << model_file_name << endl;
		return;
	}
	if(svm_check_probability_model(model) != 0)
	{
		this->is_predict_probability = true;
	}
}


SVMDetector::~SVMDetector()
{
	if(model != NULL)
		delete model;
	model = NULL;
}



int SVMDetector::SVMPredict(vector<float> & descript_vector, double *prob_est) // prob_est is a pointer to array
{
	if(this->isTest)
		return 1;

	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;
	int j;

	int i = 0;
	double target_label, predict_label;
	char *idx, *val, *label, *endptr;
	int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

	int n = 1;
	while(n < descript_vector.size())
		n *= 2;
	struct svm_node* x = (struct svm_node *) malloc(n*sizeof(struct svm_node));

	for(int i = 0; i < descript_vector.size(); i++)
	{
		x[i].index = i + 1;
		x[i].value = descript_vector[i];
	}
	x[descript_vector.size()].index = -1;
	if(this->is_predict_probability && prob_est != NULL)
		predict_label = svm_predict_probability(model, x, prob_est);
	else
		predict_label = svm_predict(model,x);
	if(!this->is_predict_probability && prob_est != NULL)
		*prob_est = -1;
	free(x);
	return predict_label;
}

void SVMDetector::detectInRectUsingOpenCVHog(const Mat & img, vector<Rect> & outputRects, vector<double> &scores, double zoom_scale, int zoom_times)
{
	Mat img_zoom = img.clone();
	HOGDescriptor hog(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	int times = 0;
	float zoom = 1.0;
	outputRects.clear();
	for (int i = 0; i < zoom_times * 1; i++)
		zoom /= zoom_scale;


	while (times < zoom_times * 3)
	{
		resize(img, img_zoom, Size(int(img.cols * zoom), int(img.rows * zoom)));

		vector<float> descriptors;
		Size winSize(16, 16);
		Size nWin((img_zoom.cols - 64) / winSize.width + 1, (img_zoom.rows - 64) / winSize.height + 1);
		hog.compute(img_zoom, descriptors, Size(16, 16));

		vector<float>::iterator ite_k = descriptors.begin();
		for (int i = 0; i <= (int)img_zoom.cols - 64; i += winSize.width)
		{

#pragma omp parallel for 
			for (int j = 0; j <= (int)img_zoom.rows - 64; j += winSize.height)
			{
				vector<float> descriptor;
				descriptor.assign(descriptors.begin() + ((i / winSize.width) + (j / winSize.height) * nWin.width) * 1764,
					descriptors.begin() + ((i / winSize.width) + (j / winSize.height)* nWin.width + 1) * 1764);
				if (this->is_predict_probability)
				{
					double score[10];
					int result = SVMPredict(descriptor, score);
					if (result == 1)
					{
#pragma omp critical
						{
							outputRects.push_back(Rect(int(i / zoom), int(j / zoom), int(64 / zoom), int(64 / zoom)));
							scores.push_back(score[0]);
						}
					}
				}
			}
		}
		zoom *= zoom_scale;
		times++;
	}
}

