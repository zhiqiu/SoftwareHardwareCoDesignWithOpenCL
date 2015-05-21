#ifndef RECTANGLES_H
#define RECTANGLES_H
#include <vector>
#include <array>
#include <opencv2\opencv.hpp>
struct Para
{
	double sw;
	double sh;
	double th;
	double ss;
};

struct rectw
{
	cv::Rect r;
	double w;
};

bool cmp(const rectw &a, const rectw &b);
void non_max_sp(std::vector<cv::Rect_<double>> &rect, std::vector<double> &score, Para para, std::vector<cv::Rect_<double>> &drect, std::vector<double> &dscore);
void compute_mode(int i,const std::vector<std::array<double, 4> > &p, const std::vector<double> &w, const Para &param
				  , std::array<double, 4> &pmode, double &wmode);
void compute_unique_modes(const std::vector<std::array<double, 4> > &pmode, const std::vector<double> &wmode,
						  const double dthresh,const Para &param,std::vector<std::array<double, 4> > &umode, std::vector<double> &uscore);

#endif
