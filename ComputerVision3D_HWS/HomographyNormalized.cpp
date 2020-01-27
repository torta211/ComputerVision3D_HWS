#include "HomographyNormalized.h"

void normalizeCoords(std::vector<cv::Point2f>& points_a, std::vector<cv::Point2f>& points_b, cv::Mat& T1, cv::Mat& T2)
{
	//points_a.size() is assumed to be equal to points_b.size()
	int num_pointpairs = points_a.size();

	float a_x_mean = 0.0f;
	float a_y_mean = 0.0f;
	float b_x_mean = 0.0f;
	float b_y_mean = 0.0f;
	float a_x_var = 0.0f;
	float a_y_var = 0.0f;
	float b_x_var = 0.0f;
	float b_y_var = 0.0f;
	for (int i = 0; i < num_pointpairs; i++)
	{
		a_x_mean += points_a[i].x;
		a_y_mean += points_a[i].y;
		b_x_mean += points_b[i].x;
		b_y_mean += points_b[i].y;
		a_x_var += points_a[i].x * points_a[i].x;
		a_y_var += points_a[i].y * points_a[i].y;
		b_x_var += points_b[i].x * points_b[i].x;
		b_y_var += points_b[i].y * points_b[i].y;
	}
	a_x_mean /= num_pointpairs;
	a_y_mean /= num_pointpairs;
	b_x_mean /= num_pointpairs;
	b_y_mean /= num_pointpairs;
	a_x_var = a_x_var / num_pointpairs - a_x_mean * a_x_mean;
	a_y_var = a_y_var / num_pointpairs - a_y_mean * a_y_mean;
	b_x_var = b_x_var / num_pointpairs - b_x_mean * b_x_mean;
	b_y_var = b_y_var / num_pointpairs - b_y_mean * b_y_mean;
	
	// Translate center of mass to origin
	cv::Mat Tr1 = cv::Mat::eye(3, 3, CV_32F);
	Tr1.at<float>(0, 2) = -1 * a_x_mean;
	Tr1.at<float>(1, 2) = -1 * a_y_mean;

	cv::Mat Tr2 = cv::Mat::eye(3, 3, CV_32F);
	Tr2.at<float>(0, 2) = -1 * b_x_mean;
	Tr2.at<float>(1, 2) = -1 * b_y_mean;

	double SQRT2 = sqrt(2);
	// Scale average point distance to sqrt(2)
	cv::Mat Sc1 = cv::Mat::eye(3, 3, CV_32F);
	Sc1.at<float>(0, 0) = SQRT2 / sqrt(a_x_var);
	Sc1.at<float>(1, 1) = SQRT2 / sqrt(a_y_var);

	cv::Mat Sc2 = cv::Mat::eye(3, 3, CV_32F);
	Sc2.at<float>(0, 0) = SQRT2 / sqrt(b_x_var);
	Sc2.at<float>(1, 1) = SQRT2 / sqrt(b_y_var);
	
	T1 = Sc1 * Tr1;
	T2 = Sc2 * Tr2;

	// Apply to data points
	for (int i = 0; i < num_pointpairs; i++)
	{
		points_a[i].x = SQRT2 * (points_a[i].x - a_x_mean) / sqrt(a_x_var);
		points_a[i].y = SQRT2 * (points_a[i].y - a_y_mean) / sqrt(a_y_var);
		points_b[i].x = SQRT2 * (points_b[i].x - b_x_mean) / sqrt(b_x_var);
		points_b[i].y = SQRT2 * (points_b[i].y - b_y_mean) / sqrt(b_y_var);
	}
}

cv::Mat estHomographyMatrix(std::vector<cv::Point2f> points_a, std::vector<cv::Point2f> points_b)
{
	//points_a.size() is assumed to be equal to points_b.size()
	cv::Mat A(2 * points_a.size(), 9, CV_32F);

	for (int i = 0; i < points_a.size(); i++)
	{
		float x_a = points_a[i].x;
		float y_a = points_a[i].y;
		float x_b = points_b[i].x;
		float y_b = points_b[i].y;

		A.at<float>(2 * i, 0) = x_a;
		A.at<float>(2 * i, 1) = y_a;
		A.at<float>(2 * i, 2) = 1.0f;
		A.at<float>(2 * i, 3) = 0.0f;
		A.at<float>(2 * i, 4) = 0.0f;
		A.at<float>(2 * i, 5) = 0.0f;
		A.at<float>(2 * i, 6) = -1 * x_b * x_a;
		A.at<float>(2 * i, 7) = -1 * x_b * y_a;
		A.at<float>(2 * i, 8) = -1 * x_b;

		A.at<float>(2 * i + 1, 0) = 0.0f;
		A.at<float>(2 * i + 1, 1) = 0.0f;
		A.at<float>(2 * i + 1, 2) = 0.0f;
		A.at<float>(2 * i + 1, 3) = x_a;
		A.at<float>(2 * i + 1, 4) = y_a;
		A.at<float>(2 * i + 1, 5) = 1.0f;
		A.at<float>(2 * i + 1, 6) = -1 * y_b * x_a;
		A.at<float>(2 * i + 1, 7) = -1 * y_b * y_a;
		A.at<float>(2 * i + 1, 8) = -1 * y_b;
	}

	cv::Mat eigen_vecs(9, 9, CV_32F);
	cv::Mat eigen_vals(9, 9, CV_32F);
	eigen(A.t() * A, eigen_vals, eigen_vecs);

	cv::Mat H(3, 3, CV_32F);
	for (int i = 0; i < 9; i++)
	{
		H.at<float>(i / 3, i % 3) = eigen_vecs.at<float>(8, i);
	}

	// Normalize
	return H * (1.0 / H.at<float>(2, 2));
}

void transformImage(cv::Mat origImg, cv::Mat& newImg, cv::Mat tr, bool isPerspective)
{
	cv::Mat invTr = tr.inv();
	const int origWIDTH = origImg.cols;
	const int origHEIGHT = origImg.rows;
	const int WIDTH = newImg.cols;
	const int HEIGHT = newImg.rows;

	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			cv::Mat newLoc(3, 1, CV_32F);
			newLoc.at<float>(0, 0) = x;
			newLoc.at<float>(1, 0) = y;
			newLoc.at<float>(2, 0) = 1.0;

			cv::Mat origLoc = invTr * newLoc;
			if (isPerspective) origLoc = (1.0 / origLoc.at<float>(2, 0)) * origLoc;

			int origX = round(origLoc.at<float>(0, 0));
			int origY = round(origLoc.at<float>(1, 0));

			if (origX >= 0 && origX < origWIDTH && origY >= 0 && origY < origHEIGHT)
			{
				newImg.at<cv::Vec3b>(y, x) = origImg.at<cv::Vec3b>(origY, origX);
			}
		}
	}
}

void HomographyDriver(int config, bool swap_a_b)
{
	std::string image_a_filename = std::string("C:/EreBere/Project/ELTE/grafika/res/homoim") + std::to_string(config) + "a.jpg";
	std::string image_b_filename = std::string("C:/EreBere/Project/ELTE/grafika/res/homoim") + std::to_string(config) + "b.jpg";
	std::string points_filename = std::string("C:/EreBere/Project/ELTE/grafika/res/homoCorrespondance") + std::to_string(config) + ".txt";
	std::cout << "reading in: " << image_a_filename << "\n";
	std::cout << "reading in: " << image_b_filename << "\n";
	std::cout << "reading in: " << points_filename << "\n";
	cv::Mat image_a = cv::imread(image_a_filename);
	cv::Mat image_b = cv::imread(image_b_filename);
	std::ifstream points_file(points_filename);
	std::vector<cv::Point2f> points_a;
	std::vector<cv::Point2f> points_b;
	std::string line;
	int x, y, next_space;
	const std::string delimiter = " ";
	while (!points_file.eof())
	{
		std::getline(points_file, line);

		next_space = line.find(delimiter);
		x = std::stoi(line.substr(0, next_space).c_str());
		line = line.substr(next_space + 1);

		next_space = line.find(delimiter);
		y = std::stoi(line.substr(0, next_space).c_str());
		line = line.substr(next_space + 1);

		std::cout << "correnspondance 1 : [" << x << ";" << y << "] - [";
		points_a.push_back(cv::Point2f(x, y));

		next_space = line.find(delimiter);
		x = std::stoi(line.substr(0, next_space).c_str());
		line = line.substr(next_space + 1);

		y = std::stoi(line.c_str());

		std::cout << x << ";" << y << "]\n";
		points_b.push_back(cv::Point2f(x, y));
	}

	if (swap_a_b)
	{
		cv::Mat img_tmp = image_a;
		std::vector<cv::Point2f> points_tmp = points_a;
		image_a = image_b;
		points_a = points_b;
		image_b = img_tmp;
		points_b = points_tmp;
	}

	// normalize the points coordinates, and keep the normalization matrices
	cv::Mat T1, T2;
	normalizeCoords(points_a, points_b, T1, T2);
	
	// calculate a transformation matrix for the normalized coordinates, then transform it to the original coordinates
	cv::Mat homoMatNorm = estHomographyMatrix(points_a, points_b);
	cv::Mat homoMat = T2.inv() * homoMatNorm * T1;
	std::cout << "\nMatrix of homography:\n" << homoMat;

	// apply the homography: transforms image_a to image_b's plane and put the result on image_b  
	transformImage(image_a, image_b, homoMat, true);
	
	cv::namedWindow("Result", CV_WINDOW_NORMAL);
	cv::imshow("Result", image_b);
	cvWaitKey(0);
	cvDestroyWindow("Result");
}
