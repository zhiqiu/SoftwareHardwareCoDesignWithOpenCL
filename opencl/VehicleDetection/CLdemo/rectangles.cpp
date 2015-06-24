#include "rectangles.h"

bool cmp(const rectw &a, const rectw &b)
{
	return a.w > b.w;
}

void non_max_sp(std::vector<cv::Rect_<double>> &rect, std::vector<double> &scores, Para param, std::vector<cv::Rect_<double>> &drect, std::vector<double> &dscores)
{
	assert(rect.size() == scores.size());

	std::vector<int> indx;
	for(int i = 0; i < scores.size(); i++)
	{
		if(scores[i] > param.th)
			indx.push_back(i);
	}
	if(indx.size() == 1)
	{
		drect.push_back(rect[indx[0]]);
		dscores.push_back(scores[indx[0]]);
	}
	else if(indx.size() == 0)
	{
		return;
	}

	double dthresh = 1e-2;
	double minw, minh;
	minw = rect[indx[0]].width;
	for(int i = 1; i < indx.size(); i++)
		minw = (rect[indx[i]].width < minw) ? rect[indx[i]].width : minw;
	minh = rect[indx[0]].height;
	for(int i = 1; i < indx.size(); i++)
		minh = (rect[indx[i]].height < minh) ? rect[indx[i]].height : minh;

	param.sw = param.sw*minw;
    param.sh = param.sh*minh;

	std::vector<std::array<double, 4> > p;
	std::vector<double> w;

	for(int i = 0; i < indx.size(); i++)
	{
		std::array<double, 4> item;
		item[0] = rect[indx[i]].x + rect[indx[i]].width / 2.0;
		item[1] = rect[indx[i]].y + rect[indx[i]].height / 2.0;
		item[2] = log(rect[indx[i]].width / minw);
		item[3] = log(rect[indx[i]].height / minh);
		p.push_back(item);
		w.push_back(scores[indx[i]] - param.th);
	}

	std::vector<std::array<double, 4> > pmode(p.size());
	std::vector<double> wmode(w.size());

	for(int i = 0 ; i < p.size(); i++)
		 compute_mode(i,p,w,param,pmode[i], wmode[i]);
	std::vector<std::array<double, 4> > umode;
	std::vector<double> uscore;
	compute_unique_modes(pmode,wmode,dthresh,param,umode,uscore);
	// sw = exp(umode(:,3))*minw;
    // sh = exp(umode(:,4))*minh;
	std::vector<double> sw, sh;
	for(int i = 0 ; i < umode.size(); i++)
	{
		sw.push_back(exp(umode[i][2]) * minw);
		sh.push_back(exp(umode[i][3]) * minh);
	}
	// drect = [umode(:,1)-0.5*sw umode(:,2)-0.5*sh sw sh];
    // dscores = uscore + param.th;
	drect.clear();
	dscores.clear();
	for(int i = 0 ; i < umode.size(); i++)
	{
		drect.push_back(cv::Rect_<double>(umode[i][0] - 0.5*sw[i], umode[i][1] - 0.5*sh[i], sw[i], sh[i]));
	}
	for(int i = 0 ; i < uscore.size(); i++)
		dscores.push_back(uscore[i] + param.th);
}

void compute_mode(int i,const std::vector<std::array<double, 4> > &p, const std::vector<double> &w, const Para &param
				  , std::array<double, 4> &pmode, double &wmode)
{
	std::array<double, 4> pmode_new;
	std::vector<double> wd;
	// pmode = p(i,:);
    // wmode = w(i);
	pmode = p[i];
	wmode = w[i];
	// npts = size(p,1);
	int npts = p.size();
	//tallones = ones(npts,1);
	int *tallones = new int[npts];
	for(int i = 0; i < npts; i++)
		tallones[i] = 1;
	// vars = [param.sw*exp(p(:,3)) param.sh*exp(p(:,4)) param.ss*tallones param.ss*tallones];
	// vars = vars.^2;
	std::vector<std::array<double, 4> > vars;
	for(int j = 0; j < npts; j++)
	{
		std::array<double, 4> item;
		item[0] = param.sw * exp(p[j][2]);
		item[1] = param.sh * exp(p[j][3]);
		item[2] = param.ss * tallones[j];
		item[3] = param.ss * tallones[j];
		item[0] *= item[0];
		item[1] *= item[1];
		item[2] *= item[2];
		item[3] *= item[3];
		vars.push_back(item);
	}

	while(1)
	{
		// d = p - repmat(pmode,[npts 1]); 
		// d = d.^2;
		std::vector<std::array<double, 4> > d;
		for(int j = 0; j < npts; j++)
		{
			std::array<double, 4> item;
			for(int k = 0; k < 4; k++)
			{
				item[k] = p[j][k] - pmode[k];
				item[k] *= item[k];
			}
			d.push_back(item);
		}

		// wd = w.*exp(-sum(d./vars,2));
		wd.clear();
		for(int j = 0; j < npts; j++)
		{
			double sum = 0;
			for(int k = 0; k < 4; k++)
				sum += d[j][k] / vars[j][k];
			wd.push_back(w[j] * exp(-sum));
		}
		// wd = wd/sum(wd);
		double sum_wd = 0;
		for(int j = 0; j <  npts; j++)
			sum_wd += wd[j];
		for(int j = 0; j <  npts; j++)
			wd[j] = wd[j] / sum_wd;

		// pmode_new = wd'*p;
		for(int k = 0; k < 4; k++)
		{
			pmode_new[k] = 0;
			for(int j = 0 ; j < npts; j++)
			{
				pmode_new[k] += wd[j] * p[j][k];
			}
		}

		// mean(abs(pmode_new-pmode)
		double mean;
		double sum = 0;
		for(int k = 0; k < 4; k++)
			sum += abs(pmode_new[k] - pmode[k]);
		mean = sum / 4;
		if(mean < 1e-3)
			break;
		// pmode = pmode_new;
		pmode = pmode_new;
	}
	// wmode = sum(w.*wd);
	wmode = 0;
	for(int j = 0; j < npts; j++)
		wmode += w[j] * wd[j];

	delete [] tallones;
}

void compute_unique_modes(const std::vector<std::array<double, 4> > &pmode, const std::vector<double> &wmode,
						  const double dthresh,const Para &param,std::vector<std::array<double, 4> > &umode, std::vector<double> &uscore)
{
	// npts = size(pmode,1);
	int npts = pmode.size();
	// tallones = ones(npts,1);
	std::vector<double> tallones(npts);
	for(int i = 0 ; i < npts; i++)
		tallones[i] = 1;
	// all=1:npts;
	std::vector<int> all;
	for(int i = 0 ; i < npts; i++)
		all.push_back(i);
	// uniq=[];
	std::vector<int> uniq;

	while(!all.empty())
	{
		// i=all(1);
		int i=all[0];
		// uniq = [uniq i];
		uniq.push_back(i);
		// d = pmode(all,:) - repmat(pmode(i,:),size(all,2),1);
		// d = mean(abs(d),2);
		std::vector<double> d;
		for(int j = 0 ; j < all.size(); j++)
		{
			double sum = 0;
			for(int k =0 ; k < 4; k++)
				sum += abs(pmode[all[j]][k] - pmode[i][k]);
			d.push_back(sum / 4);
		}
		// samei=d<thresh;
		std::vector<int> new_all;
		for(int i = 0 ;i  < all.size(); i++)
		{
			if(d[i] >= dthresh)
				new_all.push_back(all[i]);
		}
		all = new_all;
	}
	// umode = pmode(uniq,:);
    // uscore = wmode(uniq,:);
	for(int i = 0; i < uniq.size(); i++)
	{
		umode.push_back(pmode[uniq[i]]);
		uscore.push_back(wmode[uniq[i]]);
	}
}