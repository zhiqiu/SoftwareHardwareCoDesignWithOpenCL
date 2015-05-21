#include <iostream>
#include <string>
#include <fstream>

#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "SVMDetector.h"
#include "rectangles.h"

using namespace std;
using namespace cv;


int main()
{
	SVMDetector svmDetect;
	svmDetect.initDetector("vehicle_detect.model");

	Para param;
	param.sw = 0.2; //不变
	param.sh = 0.2; //不变
	param.ss = 1.3; //不变
	param.th = 0.9999; //可以改成0.9 0.95 0.99
	//param.th = 0.9;
	
	ifstream filelist("vehicle_data/filelist.txt");
	while(filelist)
	{
		string line;
		getline(filelist, line);
		if(line.size() == 0)
			break;

		Mat img = imread("vehicle_data/" + line);

		Mat img_g = img.clone();
		Mat img_r = img.clone();

		vector<Rect> rects;
		vector<double> scores;

		vector<Rect_<double> > double_rects;
		vector<Rect_<double> > drects; // 最终目标所在位置
		vector<double> dscores;
		vector<rectw> rectws; // rectws用来排序

	
		svmDetect.detectInRectUsingOpenCVHog(img, rects, scores); // 做detection

		for(int i = 0 ; i < rects.size(); i++)
		{
			rectw temp;
			temp.r = rects[i];
			temp.w = scores[i];
			rectws.push_back(temp);
		}

		sort(rectws.begin(), rectws.end(), cmp);
		scores.clear();

		for(int i = 0 ; i < rects.size(); i++)
		{
			double_rects.push_back(Rect_<double>(rectws[i].r));   //将vector<Rect>转换为vector<Rect_<double> >
			scores.push_back(rectws[i].w);
		}
			
		non_max_sp(double_rects, scores, param, drects, dscores); //聚类,NMS

		for(int i = 0 ; i < rects.size(); i++)
		{
			rectangle(img_g, rects[i], Scalar(0, 255, 0)); //画绿框
		}
		for(int i = 0 ;i < drects.size(); i++)
		{
			rectangle(img_r, drects[i], Scalar(0, 0, 255), 2);  //画最后的框
		}
		imwrite(string("test_result") + "/red/" + line, img_r);
		imwrite(string("test_result") + "/green/" + line, img_g);

		cout << line << endl;
	}
	return 0;
}
