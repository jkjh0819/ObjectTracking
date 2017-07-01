#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>

#define SEARCH_RANGE 30
#define Sampling 5
#define vector_size 15
#define Hue_impo 0.7
#define Sat_impo 0.2
#define Val_impo 0.1
#define bin 18

IplImage *img, *img2;

CvPoint LU_Point;
CvPoint RD_Point;
CvPoint Center;

int Rectangle_width;
int Rectangle_height;
bool IsRectSet = false;
bool IsModelHistogram = false;
bool IsPointSet = false;
bool back_img = true;

double histogram[bin] = { 0, };
double next_frame_histogram[bin] = { 0, };

void reset_next_frame_histogram();
void on_mouse(int event, int x, int y, int flags, void* param);
void make_model_histogram(CvPoint LU, CvPoint RD);
int get_index(int sum);
double get_model_histogram_value(int x, int y);
double get_w(int x, int y);

int min(int x, int y)
{
	return x > y ? y : x;
}
int max(int x, int y)
{
	return x > y ? x : y;
}

int main()
{
	
	CvCapture *video;
	CvPoint LU, RD;
	int key, height, width;
	double total_w = 0, w;

	cvNamedWindow("video", CV_WINDOW_AUTOSIZE);
	video = cvCaptureFromAVI("video2.avi");
	img = cvQueryFrame(video);


	do
	{
		img2 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
		cvCvtColor(img, img2, CV_BGR2HSV);
		
		if(!IsRectSet)
			cv::setMouseCallback("video", on_mouse, NULL);
		
		if (IsRectSet)
		{
			if (!IsPointSet)
			{
				LU = LU_Point;
				RD = RD_Point;
				Center.x = (LU.x + RD.x) / 2;
				Center.y = (LU.y + RD.y) / 2;
				IsPointSet = true;
			}
			if (!back_img)
			{
				cvRectangle(img, LU, RD, CV_RGB(255, 255, 255));
				cvRectangle(img2, LU, RD, CV_RGB(255, 255, 255));
			}
			if(!IsModelHistogram)
			{
				make_model_histogram(LU, RD);
				IsModelHistogram = true;
			}
		}

		cvShowImage("video", img);
		cvShowImage("video2", img2);

		cv::Mat Mat_img = cv::cvarrToMat(img2,true);
		cv::Vec3b * pixel = NULL;
		if (IsRectSet)
		{
			uchar hue;
			total_w = 0;

			CvPoint search_LU = { max(LU.x - SEARCH_RANGE, 0) , max(LU.y - SEARCH_RANGE, 0) };
			CvPoint search_RD = { min(RD.x + SEARCH_RANGE, Mat_img.rows) , min(RD.y + SEARCH_RANGE, Mat_img.cols) };

			double next_x = 0, next_y = 0;
			int abs_x = 0;
			int abs_y = 0;

			
			if (!back_img)
			{
				//mean shift
				do {
					next_x = 0, next_y = 0;
					search_LU = { max(LU.x - SEARCH_RANGE, 0) , max(LU.y - SEARCH_RANGE, 0) };
					search_RD = { min(RD.x + SEARCH_RANGE, Mat_img.rows) , min(RD.y + SEARCH_RANGE, Mat_img.cols) };

					for (int i = search_LU.x; i < search_RD.x; i+=Sampling)
					{
						
						for (int j = search_LU.y; j < search_RD.y; j+=Sampling)
						{
							w = get_w(i,j);	
							total_w += w;
							next_x += w*i;
							next_y += w*j;
						}
					}
					next_x = next_x / total_w;
					next_y = next_y / total_w;	//전체 w의 합으로 나눠줌. 

					abs_x = abs(Center.x - next_x);
					abs_y = abs(Center.y - next_y);
					Center.x = next_x;
					Center.y = next_y;
					LU.x = max(Center.x - Rectangle_width / 2, 0);
					LU.y = max(Center.y - Rectangle_height / 2, 0);
					RD.x = min(LU.x + Rectangle_width, Mat_img.rows);
					RD.y = min(LU.y + Rectangle_height, Mat_img.cols);
					
				} while (sqrt(std::pow(abs_x,2) + std::pow(abs_y, 2))>vector_size); 

				
			}

			if (back_img)
			{
				for (int i = 0; i < Mat_img.rows; i++)
				{
					pixel = Mat_img.ptr<cv::Vec3b>(i);
					for (int j = 0; j < Mat_img.cols; j++)
					{
						
						w = get_model_histogram_value(j, i);
						pixel[j][0] = (uchar)255*w; // 새 픽셀의 color histogram은 hue가 속하는 범위만 1이고 나머지는 0, 
												 //따라서 1*model color histogram(pixel)해주면 됨, 앞의 255는 흑백영상이 0~255값이라 상대값을 구하기 위해서임.
						pixel[j][1] = (uchar)255*w;
						pixel[j][2] = (uchar)255*w;
						
					}

				} //backprojection by one pixel
				cv::imshow("mat", Mat_img);
			}			
		}


		key = cvWaitKey(30);
		if (key == 27) break;//ESC key
		if (IsRectSet)
		img = cvQueryFrame(video);
	} while (img);

	return 0;
}

void reset_next_frame_histogram()
{
	for (int i = 0; i < bin; i++)
		next_frame_histogram[i] = 0;
}

void on_mouse(int event, int x, int y, int flags, void* param) {	
	if (event == CV_EVENT_LBUTTONDOWN) {
		LU_Point.x = x;
		LU_Point.y = y;
	}
	if (event == CV_EVENT_LBUTTONUP)
	{
		RD_Point.x = x;
		RD_Point.y = y;
		Rectangle_width = (RD_Point.x - LU_Point.x);
		Rectangle_height = (RD_Point.y - LU_Point.y);
		IsRectSet = true;
	}
}

void make_model_histogram(CvPoint LU, CvPoint RD) //히스토그램 만들기
{
	for (int i = LU.x; i < RD.x; i++)
	{
		for (int j = LU.y; j < RD.y; j++)
		{
			CvScalar s = cvGet2D(img2, j, i);
			int hue = (int)s.val[0], saturation = (int)s.val[1] , value = (int)s.val[2];
			int sum = (hue*Hue_impo +saturation*Sat_impo + value*Val_impo);
			histogram[sum / bin]++;
		}
	}
	int total = 0;
	for (int i = 0; i < bin; i++)
		total += histogram[i] * histogram[i];

	total = sqrt(total);
	for (int i = 0; i < bin; i++)
		histogram[i] /= total;
}

int get_index(int sum)
{
	return sum / bin;
}

double get_model_histogram_value(int x, int y)
{
	CvScalar s = cvGet2D(img2, y, x);
	int hue = (int)s.val[0];
	int saturation = (int)s.val[1];
	int value = (int)s.val[2];
	int sum = (hue*Hue_impo +saturation*Sat_impo + value*Val_impo);
	return histogram[get_index(sum)];
}

double get_w(int x, int y)//중심좌표에서 사각형 크기만큼 히스토그램을 구하고 모델 히스토그램하고의 유사도를 구함
{
	double w = 0;
	reset_next_frame_histogram();
	for (int i = max(x-Rectangle_width/2,0); i < min(x + Rectangle_width/2, img2->width); i++)
	{
		for (int j = max(y - Rectangle_width / 2, 0); j < min(y + Rectangle_width / 2, img2->height); j++)
		{
			CvScalar s = cvGet2D(img2, j, i);
			int hue = (int)s.val[0];
			int saturation = (int)s.val[1];
			int value = (int)s.val[2];
			int sum = (hue*Hue_impo+saturation*Sat_impo + value*Val_impo);
			next_frame_histogram[sum / bin]++;
		}
	}
	int total = 0;
	for (int i = 0; i < bin; i++)
		total += next_frame_histogram[i] * next_frame_histogram[i];
	
	total = sqrt(total);
	for (int i = 0; i < bin; i++)
		next_frame_histogram[i] /= total;

	for (int i = 0; i < bin; i++)
		w += sqrt(histogram[i] * next_frame_histogram[i]);

	return w;
}
