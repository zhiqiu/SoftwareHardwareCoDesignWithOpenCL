/*
尽量不用阈值调节+分区检测+曲线拟合+预测
*/

#include<iostream>
//#include"lsd.h"
#include"LaneDetector.h"

bool angleCMP(LaneInfo a, LaneInfo b);

int main(){

	LaneDetector ld;
	ld.setInput("D:/WorkSpace/VisualStudio/Bird_Eye/Bird_Eye/Video20111021124915174.avi");

	//图像切割，之后可用图像分割算法找出天空与地面交点，更好求出分割点
	ld.cutY = ld.originHeight / 3 - 50 ;
	ld.cutHeight = ld.originHeight / 3 ;
	//ld.cutHeight = ld.originHeight - ld.cutY;
	ld.rect1 = Rect(0, ld.cutY, ld.originWidth, ld.cutHeight);
	ld.capture >> ld.image;
	ld.image(ld.rect1).copyTo( ld.ImageCut);

	//lsd直线检测
	ld.lsdDetect(ld.ImageCut, ld.laneinfos, PERSPECTIVEMODE);
	ld.myNormalize(ld.laneinfos);
	/*Mat show = ld.drawLine(ld.ImageCut, ld.laneinfos, DRAWNORMAL);
	imshow("show",show);
	waitKey(0);*/

	//聚类
	ld.clusterNum = 0;
	ld.myBisectKmeans_Extend_Angle(ld.laneinfos, ld.SeperateLine, ld.clusterNum, 0.25);
	//cout<<"clusterNum： "<<ld.clusterNum<<endl;
	//Mat clusterShow = ld.drawLine(ld.ImageCut, ld.SeperateLine, ld.clusterNum);
	ld.getAbstractLane(ld.SeperateLine, ld.abstractLane, ld.clusterNum);
	//Mat abstrShow =  ld.drawLine(ld.ImageCut, ld.abstractLane);
	sort(ld.abstractLane.begin(), ld.abstractLane.end(), angleCMP);

	//IPM
	ld.ipmLeft = 400, ld.ipmRight = 500;
	Point2f origin[4],dest[4];
	origin[0] = ld.abstractLane[0].extendPt1; origin[1] = ld.abstractLane[0].extendPt2;
	origin[2] = ld.abstractLane[1].extendPt1; origin[3] = ld.abstractLane[1].extendPt2;
	dest[0] = Point2f(ld.ipmLeft,0); dest[1] = Point2f(ld.ipmLeft,ld.cutHeight);
	dest[2] = Point2f(ld.ipmRight,0); dest[3] = Point2f(ld.ipmRight,ld.cutHeight);

	ld.WarpMatrix= Mat::zeros(3,3,CV_32FC1);
	ld.WarpMatrix = getPerspectiveTransform(origin,dest);
	ld.ImageIPM = Mat::zeros(ld.ImageCut.cols,ld.ImageCut.rows,ld.ImageCut.type());
	warpPerspective(ld.ImageCut,ld.ImageIPM,ld.WarpMatrix,Size(ld.ImageCut.cols,ld.ImageCut.rows));

	//分段
	for(int i=0; i<PARTITION; i++){
		ld.HeightPartition[i] = i *ld.cutHeight / 4;
		line(ld.ImageCut, Point(0,ld.HeightPartition[i]), Point(ld.originWidth, ld.HeightPartition[i]), Scalar(255), 3);
		Rect rect(0, ld.HeightPartition[i], ld.originWidth, ld.cutHeight / 4);
		ld.ImageIPM(rect).copyTo(ld.ImageIPMPartition[i]);
	}
	
	//求出逆矩阵
	ld.InverseWarpMatrix = Mat::zeros(3,3,CV_32FC1);
	invert(ld.WarpMatrix, ld.InverseWarpMatrix);

	//第一次检测
	//对PARTITION个区间进行检测，找到直线，并将其中点放入某个结构中储存
	for(int i=0; i<PARTITION; i++){

		ld.lsdDetect(ld.ImageIPMPartition[i], ld.lanePartition[i], BIRDEYEMODE);
		ld.myNormalize(ld.lanePartition[i]);
		//Mat show1 = ld.drawLine(ld.ImageIPMPartition[i], ld.lanePartition[i], DRAWNORMAL);
		//cout<<"ld.lanePartition[i] num: "<<ld.lanePartition[i].size()<<endl;

		//按Pt1.x聚类
		ld.partitionClusterNum[i] = 0;
		ld.myBisectKmeans_PtX(ld.lanePartition[i], ld.lanePartitionSeperate[i], ld.partitionClusterNum[i], 10);
		//Mat show = ld.drawLine(ld.ImageIPMPartition[i], ld.lanePartitionSeperate[i], ld.partitionClusterNum[i], DRAWNORMAL);
		/*int seCount = 0;
		for(int m=0; m<ld.partitionClusterNum[i]; m++){
			for(int j=0; j<ld.lanePartitionSeperate[i][m].size(); j++){
				seCount++;
			}
		}*/
		//cout<<"seCount: "<<seCount<<endl;

		ld.getAbstractLane(ld.lanePartitionSeperate[i], ld.lanePartitionAbstract[i], ld.partitionClusterNum[i]);

		for(int j=0; j<ld.lanePartitionAbstract[i].size(); j++){//区分该线属于哪一个区间
			ld.lanePartitionAbstract[i][j].partitionType = i;

			//确定出检测区域
			DetectArea da;
			da.startX = ld.lanePartitionAbstract[i][j].midPoint.x - 30;
			da.endX = ld.lanePartitionAbstract[i][j].midPoint.x + 30;
			da.partitionType = i;
			da.length = 60;
			da.height = 100; //该处可能产生bug，要注意
			ld.detectArea[i].push_back(da);
		}	
		//Mat PartitionShow = ld.drawLine(ld.ImageIPMPartition[i], ld.lanePartitionAbstract[i], DRAWEXTEND);

		//将检测出的线的中点投影，进行贝塞尔曲线拟合	
		//vector<Point> points = ld.LaneProject(i); //傻逼了...不应该这样做
		//Mat test = ld.drawPoints(ld.image,points);

		//imshow("test",test);
		//imshow("show",show);
		//imshow("show1",show1);
		//imshow("ImageIPMPartition", ld.ImageIPMPartition[i]);
		//imshow("lsdShowPartition",lsdShowPartition);
		//imshow("PartitionShow",PartitionShow);
		//imshow("lsd",lsd);
		//waitKey(0);
	}

	//Show Test
	/*for(int i=0; i<PARTITION; i++){
		for(int j=0; j<ld.detectArea[i].size(); j++){
			Rect rect(ld.detectArea[i][j].startX, 0, ld.detectArea[i][j].length ,100);
			Mat test = Mat::zeros(60, 100, ld.ImageIPMPartition[i].type()) ;
			ld.ImageIPMPartition[i](rect).copyTo(test);
			imshow("test",test);
			waitKey(0);
		}
	}*/


	//将得到的各区线段进行分类
	//先将第一块的线排好；
	ld.sameLaneNum = 0;
	for(int i=0; i<ld.lanePartitionAbstract[0].size(); i++){
		ld.lanePartitionAbstract[0][i].clusterType = ld.sameLaneNum++;
		ld.sameLane[i].push_back(ld.lanePartitionAbstract[0][i]);
	}
	//再排剩余的块，分类
	for(int i=1; i<PARTITION; i++){
		for(int j=0; j<ld.lanePartitionAbstract[i].size(); j++){//对于每一条在partition中的线
			bool flag = false;
			for(int k=0; k<ld.lanePartitionAbstract[i-1].size(); k++){
				if(abs(ld.lanePartitionAbstract[i][j].extendPt1.x - ld.lanePartitionAbstract[i-1][k].extendPt1.x) < 10 ){//则认为他们是一类
					//cout<<"clusterType"<<ld.lanePartitionAbstract[i-1][k].clusterType<<endl;
					ld.sameLane[ld.lanePartitionAbstract[i-1][k].clusterType].push_back(ld.lanePartitionAbstract[i][j]);
					ld.lanePartitionAbstract[i][j].clusterType = ld.lanePartitionAbstract[i-1][k].clusterType;//这个必不可少
					flag = true;
					/*Mat test;
					ld.ImageIPMPartition[i].copyTo(test);
					line(test, ld.lanePartitionAbstract[i][j].extendPt1, ld.lanePartitionAbstract[i][j].extendPt2, Scalar(255), 3);
					imshow("normal_lane",test);
					waitKey(0);*/
					break;
				}
			}
			if(!flag){
				ld.sameLane[ld.sameLaneNum].push_back(ld.lanePartitionAbstract[i][j]);
				ld.lanePartitionAbstract[i][j].clusterType = ld.sameLaneNum++;
				/*Mat test;
				ld.ImageIPMPartition[i].copyTo(test);
				line(test, ld.lanePartitionAbstract[i][j].extendPt1, ld.lanePartitionAbstract[i][j].extendPt2, Scalar(255), 3);
				imshow("AdditionLane",test);
				waitKey(0);*/
			}
		}
	}
	
	//Show Test
	/*Mat sameTest = Mat::zeros(ld.ImageIPM.rows, ld.ImageIPM.cols, ld.ImageIPM.type() );
	for(int i=0; i<ld.sameLaneNum; i++){
		int r = rand()%256, g = rand()%256, b = rand()%256;
		for(int j=0; j<ld.sameLane[i].size(); j++){
			line(sameTest, Point(ld.sameLane[i][j].extendPt1.x, ld.sameLane[i][j].extendPt1.y + 
				ld.HeightPartition[ld.sameLane[i][j].partitionType]), 
				Point(ld.sameLane[i][j].extendPt2.x, ld.sameLane[i][j].extendPt2.y + 
				ld.HeightPartition[ld.sameLane[i][j].partitionType]), Scalar(r,g,b),3);
		}
	}
	imshow("sameTest",sameTest);
	waitKey(0);*/
	//将分好的线投影回去，投影参数：extend.x extend.y mid.x mid.y
	for(int i=0; i<ld.sameLaneNum; i++){
		//cout<<"same lane: "<<ld.sameLane[i].size()<<endl;
		ld.projectLane[i] = ld.LaneProject(ld.sameLane[i]);
		//Mat test = ld.drawLine(ld.image, ld.projectLane[i],DRAWEXTEND);
		//imshow("test",test);
		//waitKey(0);
	}
	
	//Show Test
	//Mat test = ld.drawLine(ld.image, ld.projectLane, ld.sameLaneNum ,DRAWEXTEND);
	//imshow("test",test);

	//Bezier曲线
	Mat result ;
	ld.image.copyTo(result);
	for(int i=0; i<ld.sameLaneNum; i++){
		int size = ld.projectLane[i].size();
		if(size == 4){
			Point3f points[4];
			for(int m=0; m<ld.projectLane[i].size(); m++){
				points[m].x = ld.projectLane[i][m].midPoint.x;
				points[m].y = ld.projectLane[i][m].midPoint.y;
				circle(result,ld.projectLane[i][m].midPoint, 10, Scalar(255), -1);
			}
			ld.ecvDrawBezier(result,points);
		}
		else if(size > 1){//若检测到的线段数少于四且大于一，则用最小二乘拟合
			vector<Point> points;
			Vec4f lines;
			for(int m=0; m<size; m++){
				points.push_back(ld.projectLane[i][m].midPoint);
			}
			fitLine(points, lines, CV_DIST_L2,1, 0.001, 0.001);
			Vec4f ConvertLine;
			float k = lines[1]/lines[0];
			float b = lines[3] - k * lines[2];
			ConvertLine[0] = (ld.image.cols - b) / (lines[1]/lines[0]);
			ConvertLine[1] = ld.image.cols;
			ConvertLine[3] = ld.cutY;
			ConvertLine[2] = (ConvertLine[3]-b) / k;	
			line(result, Point(ConvertLine[0],ConvertLine[1]),Point(ConvertLine[2],ConvertLine[3]), Scalar(255), 3);
		}
		else{	
			//只有一条直线
			line(result, ld.projectLane[i][0].extendPt1, ld.projectLane[i][0].extendPt2, Scalar(255), 3);
		}
	}
	imshow("ShowImage",result);
	imshow("IPM",ld.ImageIPM);
	//imshow("abstrShow",abstrShow);
	//imshow("lsd",lsdShow);
	imshow("ImageCut",ld.ImageCut);
	//imshow("clusterShow",clusterShow);
	waitKey(0);

	//找出可能检测区域

	//开始常规检测： IPM中分区图加小块检测
	while(true){
		//剪切图片
		ld.capture >> ld.image;
		ld.image(ld.rect1).copyTo( ld.ImageCut);
		//IPM
		warpPerspective(ld.ImageCut,ld.ImageIPM,ld.WarpMatrix,Size(ld.ImageCut.cols,ld.ImageCut.rows));
		//分段
		for(int i=0; i<PARTITION; i++){
			ld.HeightPartition[i] = i *ld.cutHeight / 4;
			line(ld.ImageCut, Point(0,ld.HeightPartition[i]), Point(ld.originWidth, ld.HeightPartition[i]), Scalar(255), 3);
			Rect rect(0, ld.HeightPartition[i], ld.originWidth, ld.cutHeight / 4);
			ld.ImageIPM(rect).copyTo(ld.ImageIPMPartition[i]);
			//初始化
			ld.lanePartition[i].clear();
		}
		//分区并检测
		for(int i=0; i<PARTITION; i++){
			int detectNum = ld.detectArea[i].size();
			for(int j=0; j<detectNum; j++){
				Rect rect(ld.detectArea[i][j].startX, 0, ld.detectArea[i][j].length, ld.detectArea[i][j].height);
				Mat detectAreaImage ; 
				ld.ImageIPMPartition[i](rect).copyTo(detectAreaImage);
				ld.lsdDetect(detectAreaImage, ld.lanePartition[i], BIRDEYEMODE);
				ld.myNormalize(ld.lanePartition[i]);
				Mat test = ld.drawLine(detectAreaImage, ld.lanePartition[i], DRAWEXTEND);
				imshow("detectAreaImage",detectAreaImage);
				imshow("detect lines", test);
				cout<<"Detect Line Number: "<<ld.lanePartition[i].size()<<endl;
				waitKey(0);
				destroyAllWindows();
			}
		}
	}



	
	return 0;
}

bool angleCMP(LaneInfo a, LaneInfo b){
	return (90-abs(a.angle)) < (90-abs(b.angle));
}