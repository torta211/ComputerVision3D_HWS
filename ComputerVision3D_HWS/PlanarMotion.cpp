#include "PlanarMotion.h"

cv::Mat fundamentalRANSAC(const std::vector<cv::Point2f>& points_1, const std::vector<cv::Point2f>& points_2)
{
	float threshold = 2.0f;
	constexpr size_t sample_size = 4; // Sample size
	size_t* const sample = new size_t[sample_size];
	std::vector<int> tmp_inliers, inliers;
	tmp_inliers.reserve(points_1.size()); // we wont have more inliers than the number of points
	cv::Mat F = cv::Mat(4, 1, CV_32F); 
	size_t iteration_number = std::numeric_limits<size_t>::max();

	for (size_t iteration = 0; iteration < iteration_number; ++iteration)
	{
		// Select a random sample of size 4
		size_t next_index_in_sample = 0;
		while (next_index_in_sample < sample_size)
		{
			size_t selected_index = rand() % points_1.size();

			bool already_in_current_sample = false;
			for (size_t in_sample_idx = 0; in_sample_idx < next_index_in_sample && !already_in_current_sample; ++in_sample_idx)
			{
				if (sample[in_sample_idx] == selected_index)
				{
					already_in_current_sample = true;
				}
			}

			if (!already_in_current_sample)
			{
				sample[next_index_in_sample] = selected_index;
				next_index_in_sample += 1;
			}
			// UGLY: undeterministic running time
		}

		// determine F for four point pairs
		// F = [0, F1, 0; F2 0 F3; 0 F4 0] matrixequation: [x2, y2, 1] * F * [x1; y1; 1] = [0]
		// equation [y1x2 x1y2 y2 y1]*[F1; F2; F3; F4] = 0
		cv::Mat A(4, 4, CV_32F);
		for (int i = 0; i < sample_size; i++)
		{
			float x1 = points_1[sample[i]].x;
			float y1 = points_1[sample[i]].y;
			float x2 = points_2[sample[i]].x;
			float y2 = points_2[sample[i]].y;

			A.at<float>(i, 0) = -1 * y1 * x2;
			A.at<float>(i, 1) = x1 * y2;
			A.at<float>(i, 2) = -1 * y2;
			A.at<float>(i, 3) = y1;
		}
		cv::Mat eigen_vecs(4, 4, CV_32F);
		cv::Mat eigen_vals(4, 4, CV_32F);
		eigen(A.t()* A, eigen_vals, eigen_vecs);

		cv::Mat F_tmp = cv::Mat(3, 3, CV_32F);
		F_tmp.at<float>(0, 1) = eigen_vecs.at<float>(3, 0);
		F_tmp.at<float>(1, 0) = eigen_vecs.at<float>(3, 1);
		F_tmp.at<float>(1, 2) = eigen_vecs.at<float>(3, 2);
		F_tmp.at<float>(2, 1) = eigen_vecs.at<float>(3, 3);

		// Iterate through pointpairs and count the inliers
		tmp_inliers.resize(0);
		cv::Mat U2T = cv::Mat(1, 3, CV_32F);
		cv::Mat U1 = cv::Mat(3, 1, CV_32F);
		for (size_t pair_idx = 0; pair_idx < points_1.size(); ++pair_idx)
		{
			U2T.at<float>(0, 0) = points_2[pair_idx].x;
			U2T.at<float>(0, 1) = points_2[pair_idx].y;
			U2T.at<float>(0, 2) = 1.0f;

			U1.at<float>(0, 0) = points_1[pair_idx].x;
			U1.at<float>(1, 0) = points_1[pair_idx].y;
			U1.at<float>(2, 0) = 1.0f;

			cv::Mat val = U2T * F_tmp * U1;
			std::cout << "val mat = " << val << "\n";
			
			if (abs(val.at<float>(0,0)) <= threshold) { tmp_inliers.push_back(pair_idx); }
		}
		std::cout << "current size of inliers: " << tmp_inliers.size() << ", so far best size: " << inliers.size() << std::endl;

		// If the current sphere has more inliers than the previous so-far-the-best, update the best parameters
		if (tmp_inliers.size() > inliers.size())
		{
			tmp_inliers.swap(inliers);
			F = F_tmp;
			std::cout << "Iteration number before update: " << iteration_number << ", ";
			iteration_number = getIterationNumber(0.9, inliers.size(), points_1.size(), 4);
			std::cout << "Iteration number after update: " << iteration_number << std::endl;
		}

	}//----end of iterations loop

	std::cout << "Number of iterations to do has been reached." << std::endl;
	// Clean up the memory
	delete[] sample;

	return F;
}

void planarMotionDriver()
{
	// Intrinsic camera parameters from file
	float cx = 517.12973, cy = 395.59665, fx = 795.11588, fy = 795.11588, focal_length = 0.002f;

	//setting random seed
	srand(time(NULL));

	cv::Mat K = cv::Mat(3, 3, CV_32F);

	K.at<float>(0, 0) = fx;
	K.at<float>(0, 1) = 0;
	K.at<float>(0, 2) = cx;
	K.at<float>(1, 0) = 0;
	K.at<float>(1, 1) = fy;
	K.at<float>(1, 2) = cy;
	K.at<float>(2, 0) = 0;
	K.at<float>(2, 1) = 0;
	K.at<float>(2, 2) = 1;

	// Load result of online ASIFT
	std::vector<cv::Point2f> points_1;
	std::vector<cv::Point2f> points_2;
	std::string filename = "C:/EreBere/Project/ELTE/grafika/res/match_ASIFT.txt";
	std::cout << "reading in: " << filename << "\n";
	std::ifstream correspondace_file(filename);
	std::string line;
	float x1, y1, x2, y2;
	int next_space;
	const std::string delimiter = " ";
	while (!correspondace_file.eof())
	{
		std::getline(correspondace_file, line);

		next_space = line.find(delimiter);
		x1 = std::stof(line.substr(0, next_space).c_str());
		line = line.substr(next_space + 1);

		next_space = line.find(delimiter);
		y1 = std::stof(line.substr(0, next_space).c_str());
		line = line.substr(next_space + 1);

		points_1.push_back(cv::Point2f(x1, y1));

		next_space = line.find(delimiter);
		x2 = std::stof(line.substr(0, next_space).c_str());
		line = line.substr(next_space + 1);

		next_space = line.find(delimiter);
		y2 = std::stof(line.substr(0, next_space).c_str());

		points_2.push_back(cv::Point2f(x2, y2));
	}

	// normalize these points
	cv::Mat T1, T2;
	normalizeCoords(points_1, points_2, T1, T2);

	// estimate F matrix
	cv::Mat F = fundamentalRANSAC(points_1, points_2);

	cv::Mat E = F;
	F.at<float>(0, 1) /= focal_length * focal_length;
	F.at<float>(1, 0) /= focal_length * focal_length;
	F.at<float>(1, 2) /= focal_length;
	F.at<float>(2, 1) /= focal_length;

	float tx = E.at<float>(2, 1);
	float tz = -1 * E.at<float>(0, 1);
	float CosBminSinB = E.at<float>(1, 0) / tz;
	float CosBplusSinB = -1.0 * E.at<float>(1, 2) / tx;
	float cosB = (CosBminSinB + CosBplusSinB) / 2.0f;
	float beta = acos(cosB);

	
}
