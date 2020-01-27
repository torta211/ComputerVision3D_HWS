#include "CircleFittingRansac.h"

void drawPoints(const std::vector<cv::Point2d>& points_, cv::Mat& image_, const cv::Scalar& color_, const double& size_)
{
	for (size_t point_idx = 0; point_idx < points_.size(); ++point_idx)
	{
		circle(image_, points_[point_idx], size_, color_, -1);
	}
}

void drawInliers(const std::vector<cv::Point2d>& points_, cv::Mat& image_, const cv::Scalar& color_, const std::vector<int> inliers_, const double& size_)
{
	for (size_t inlier_idx = 0; inlier_idx < inliers_.size(); ++inlier_idx)
	{
		circle(image_, points_[inliers_[inlier_idx]], size_, color_, -1);
	}
}

size_t getIterationNumber(double confidence_, size_t inlier_number_, size_t point_number_, size_t sample_size_)
{
	const double one_minus_confidence = 1.0 - confidence_;
	const double log_confidence = log(one_minus_confidence);
	const double inlier_ratio = static_cast<double>(inlier_number_) / point_number_;
	const double pow_inlier_ratio = std::pow(inlier_ratio, sample_size_);

	return log_confidence / log(1.0 - pow_inlier_ratio);
}

// Modification of the algorithm from the practice (this time it is RANSAC for circle fitting)
// inliers and circle_params gets set
void fitCircleRANSAC(const std::vector<cv::Point2d>* const points_, std::vector<int>& inliers_, cv::Mat& circle_params_, double threshold, double confidence, cv::Mat& canvas)
{
	constexpr size_t sample_size = 3; // Sample size
	size_t* const sample = new size_t[sample_size];
	std::vector<int> tmp_inliers;
	tmp_inliers.reserve(points_->size()); // we wont have more inliers than the number of points
	circle_params_.create(3, 1, CV_64F); // we create a 3 element, double precision matrix for the parametes of the circle (cx, cy, r) where the circle is (x-cx)^2+(y-cy)^2=r^2
	size_t iterations_todo = std::numeric_limits<size_t>::max();

	cv::namedWindow("RANSAC RUNNING", cv::WINDOW_NORMAL);

	for (size_t iteration = 0; iteration < iterations_todo; ++iteration)
	{
		// Select a random sample of size 3
		size_t next_index_in_sample = 0;
		while (next_index_in_sample < sample_size)
		{
			size_t selected_index = rand() % points_->size();

			bool already_in_current_sample = false;
			for (size_t in_sample_idx = 0;
				in_sample_idx < next_index_in_sample && !already_in_current_sample;
				already_in_current_sample = sample[in_sample_idx++] == selected_index);

			if (!already_in_current_sample)
			{
				sample[next_index_in_sample] = selected_index;
				next_index_in_sample += 1;
			}
			// UGLY: undeterministic running time
		}
		std::cout << "\niteration " << iteration  << "/" << iterations_todo << ": Selected indices: " << sample[0] << ", " << sample[1] << ", " << sample[2] << "\n";

		// Fit a circle to the selected points
		double x1 = points_->at(sample[0]).x;
		double y1 = points_->at(sample[0]).y;
		double x2 = points_->at(sample[1]).x;
		double y2 = points_->at(sample[1]).y;
		double x3 = points_->at(sample[2]).x;
		double y3 = points_->at(sample[2]).y;

		double a = (y1 - y2) / (x2 - x1);
		double b = (x2 * x2 + y2 * y2 - x1 * x1 - y1 * y1) / (2 * x2 - 2 * x1);

		// some problems...
		if (x2 == x1 || -2 * x1 * a - 2 * y1 + 2 * x3 * a + 2 * y3 == 0)
		{
			iteration -= 1;
			continue; // UGLY: ... extra ugly
		}

		double cy = (x3 * x3 - 2 * x3 * b + y3 * y3 - x1 * x1 + 2 * x1 * b - y1 * y1) / (-2 * x1 * a - 2 * y1 + 2 * x3 * a + 2 * y3);
		double cx = a * cy + b;
		double r = sqrt((x1 - cx) * (x1 - cx) + (y1 - cy) * (y1 - cy));
		std::cout << "circle params: [" << cx << ", " << cy << ", " << r << "]\n";

		// Iterate through all the points and count the inliers
		tmp_inliers.resize(0);
		size_t point_idx = 0;
		int ere = 0;
		while (point_idx < points_->size())
		{
			ere += point_idx;
			double x = points_->at(point_idx).x;
			double y = points_->at(point_idx).y;
			double distance = abs(sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) - r);
			if (distance < threshold)
			{
				tmp_inliers.push_back(point_idx);
			}
			++point_idx;
		}
		std::cout << "current size of inliers: " << tmp_inliers.size() << ", so far best size: " << inliers_.size() << "\n";

		// If the current circle has more inliers than the previous so-far-the-best, update the best parameters
		if (tmp_inliers.size() > inliers_.size())
		{
			std::cout << "NEW WINNER\n";
			tmp_inliers.swap(inliers_);

			circle_params_.at<double>(0) = cx;
			circle_params_.at<double>(1) = cy;
			circle_params_.at<double>(2) = r;
			iterations_todo = getIterationNumber(confidence, inliers_.size(), points_->size(), 2);
		}

		// Lets display the result of this iteration
		cv::Mat current_image = canvas.clone();
		drawInliers(*points_, current_image, cv::Scalar(0, 255, 0), tmp_inliers, 4); // canvas already has the points we only draw inliers
		circle(current_image, cv::Point2d(points_->at(sample[0]).x, points_->at(sample[0]).y), 15, cv::Scalar(255, 0, 0), 3);
		circle(current_image, cv::Point2d(points_->at(sample[1]).x, points_->at(sample[1]).y), 15, cv::Scalar(255, 0, 0), 3);
		circle(current_image, cv::Point2d(points_->at(sample[2]).x, points_->at(sample[2]).y), 15, cv::Scalar(255, 0, 0), 3);
		circle(current_image, cv::Point2d(cx, cy), r, cv::Scalar(255, 0, 0), 3);
		cv::imshow("RANSAC RUNNING", current_image);

		cvWaitKey(0);
	}//----end of iterations loop
	cvDestroyWindow("RANSAC RUNNING");
	std::cout << "Number of iterations to do has been reached.\n";
	cvWaitKey(0);
	// Clean up the memory
	delete[] sample;
}

cv::Mat position_image(const cv::Mat& original_, int dx, int dy)
{
	std::cout << "position into: [" << dx << " " << dy << "]" << std::endl;
	cv::Mat result = original_.clone();
	for (int x = 0; x < result.cols; ++x)
	{
		for (int y = 0; y < result.rows; ++y)
		{
			if (x - dx >= 0 && y - dy >= 0 && x - dx < original_.cols && y - dy < original_.rows)
			{
				result.at<cv::Vec3b>(y, x) = original_.at<cv::Vec3b>(y - dy, x - dx);
			}
			else
			{
				result.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
			}
		}
	}
	return result;
}

void RansacCircleDriver(double th, double conf, int config)
{
	//set a random seed
	unsigned int time_ui = unsigned int(time(NULL));
	srand(time_ui);

	//the image we are going to draw on (canvas)
	std::string image_file_name = std::string("C:/EreBere/Project/ELTE/grafika/res/image") + std::to_string(config) + ".png";
	std::string data_file_name = std::string("C:/EreBere/Project/ELTE/grafika/res/points") + std::to_string(config) + ".txt";
	std::cout << "reading in: " << image_file_name << "\n";
	cv::Mat canvas = cv::imread(image_file_name);

	std::cout << "reading in: " << data_file_name << "\n";
	std::ifstream points_file(data_file_name);
	std::vector<cv::Point2d> points;
	int coord1, coord2;
	while (!points_file.eof())
	{
		points_file >> coord1;
		points_file >> coord2;
		points.push_back(cv::Point2d(coord1, coord2));
	}

	//we will position the big image with a, w, s, d
	int delta_x = 0;
	int delta_y = 0;

	//draw these to the original image
	drawPoints(points, canvas, cv::Scalar(0, 0, 255), 2);

	//lets run RANSAC fitting
	std::vector<int> inliers;
	cv::Mat circle_params;
	fitCircleRANSAC(&points, inliers, circle_params, th, conf, canvas);

	//draw the found circle
	circle(canvas, cv::Point2d(circle_params.at<double>(0), circle_params.at<double>(1)), circle_params.at<double>(2), cv::Scalar(0, 255, 0));

	//lets display the input with a loop, that handles moving in the image
	cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
	bool go_on = true;

	while (go_on)
	{
		switch (cvWaitKey(30)) {
		case 27:
			go_on = false;
			break;
		case 119: //w
			delta_y -= 100;
			break;
		case 115: //s
			delta_y += 100;
			break;
		case 97: //a
			delta_x -= 100;
			break;
		case 100: //d
			delta_x += 100;
			break;
		}
		cv::Mat image_to_show = position_image(canvas, delta_x, delta_y);

		cv::imshow("Result", image_to_show);
		cv::resizeWindow("Result", 1600, 960);
	}
	cvDestroyWindow("Result");
	std::cout << "Circle fitting with RANSAC exits.\n";
	cvWaitKey(0);
	return;
}