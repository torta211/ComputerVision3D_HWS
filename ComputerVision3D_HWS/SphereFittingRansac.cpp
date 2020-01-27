#include "SphereFittingRansac.h"

// Modification of the algorithm from the practice (this time it is RANSAC for sphere fitting)
// inliers and sphere_params gets set
void fitSphereRANSAC(const std::vector<cv::Point3d>* const points_, std::vector<int>& inliers_, cv::Mat& sphere_params_, double threshold, double confidence)
{
	constexpr size_t sample_size = 4; // Sample size
	size_t* const sample = new size_t[sample_size];
	std::vector<int> tmp_inliers;
	tmp_inliers.reserve(points_->size()); // we wont have more inliers than the number of points
	sphere_params_.create(4, 1, CV_64F); // we create a 4 element, double precision matrix for the parametes of the circle (cx, cy, cz, r) where the circle is (x-cx)^2+(y-cy)^2+(z-cz)^2=r^2
	size_t iteration_number = std::numeric_limits<size_t>::max();

	for (size_t iteration = 0; iteration < iteration_number; ++iteration)
	{
		// Select a random sample of size 4
		size_t next_index_in_sample = 0;
		while (next_index_in_sample < sample_size)
		{
			size_t selected_index = rand() % points_->size();

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
		//std::cout << "iter " << iteration << ": Selected indices: " << sample[0] << ", " << sample[1] << ", " << sample[2] << ", " << sample[3] << std::endl;

		// Fit a sphere to the selected points
		double x1 = points_->at(sample[0]).x;
		double y1 = points_->at(sample[0]).y;
		double z1 = points_->at(sample[0]).z;
		double x2 = points_->at(sample[1]).x;
		double y2 = points_->at(sample[1]).y;
		double z2 = points_->at(sample[1]).z;
		double x3 = points_->at(sample[2]).x;
		double y3 = points_->at(sample[2]).y;
		double z3 = points_->at(sample[2]).z;
		double x4 = points_->at(sample[3]).x;
		double y4 = points_->at(sample[3]).y;
		double z4 = points_->at(sample[3]).z;

		// plug each point into the equation of sphere, substract equation I from each of the others. we get 3 equations of the form: 2(x1-xi)cx + 2(y1-yi)cy + 2(z1-zi)cz + k_i_I = 0
		double k_II_I = x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1 + z2 * z2 - z1 * z1;
		double k_III_I = x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1 + z3 * z3 - z1 * z1;
		double k_IV_I = x4 * x4 - x1 * x1 + y4 * y4 - y1 * y1 + z4 * z4 - z1 * z1;

		// now with the resulting 3 equations (1, 2, 3), substract 1 from 2. we get: 2(x2-x3)cx + 2(y2-y3)cy + 2(z2-z3)cz + k_III_I - k_II_I = 0
		// cz = k_cz + k_cx*cx + k_cy*cy
		double k_cz = (k_II_I - k_III_I) / (2 * (z2 - z3));
		double k_cx = (x3 - x2) / (z2 - z3);
		double k_cy = (y3 - y2) / (z2 - z3);

		// plugging the expression of cz into equation 2, we get: (2(x1-x3)+2(z1-z3)k_cx)cx + (2(y1-y3)+2(z1-z3)k_cy)cy + K_III_I + 2(z1-z3)k_cz = 0
		// let's introduce a, b, c such that a*cx + b*cy + c = 0 -> cy = -c/b - (a/b) * cx
		double a = 2 * (x1 - x3) + 2 * (z1 - z3) * k_cx;
		double b = 2 * (y1 - y3) + 2 * (z1 - z3) * k_cy;
		double c = k_III_I + 2 * (z1 - z3) * k_cz;

		// plugging into equation 3, we will only have constants and cx: 2(x1-x4)cx + 2(y1-y4)(-c/b-(a/b)*cx) + 2(z1-z4)(k_cz+k_cx*cx+k_cy(-c/b-(a/b)cx)) + k_III_I = 0
		double numerator = -1 * (k_III_I + 2 * (y1 - y4) * (-1 * c / b) + 2 * (z1 - z4) * k_cz + 2 * (z1 - z4) * k_cy * (-1 * c / b));
		double denominator = 2 * (x1 - x4) + 2 * (y1 - y4) * (-1 * a / b) + 2 * (z1 - z4) * k_cx + 2 * (z1 - z4) * k_cy * (-1 * a / b);

		double cx = numerator / denominator;
		double cy = -1 * c / b - a / b * cx;
		double cz = k_cz + k_cx * cx + k_cy * cy;
		double r = sqrt((x1 - cx) * (x1 - cx) + (y1 - cy) * (y1 - cy) + (z1 - cz) * (z1 - cz));
		if (z3 == z2 || b == 0 || denominator == 0) // UGLY: problem handling
		{
			iteration -= 1; // lets not count NaN params as an iteration
			continue;
		}
		//std::cout << "sphere params: [" << cx << ", " << cy << ", " << cz << ", " << r << "]" << std::endl;

		// Iterate through all the points and count the inliers
		tmp_inliers.resize(0);
		for (size_t point_idx = 0; point_idx < points_->size(); ++point_idx)
		{
			double x = points_->at(point_idx).x;
			double y = points_->at(point_idx).y;
			double z = points_->at(point_idx).z;
			double distance = abs(sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz)) - r);
			if (distance <= threshold) { tmp_inliers.push_back(point_idx); }
		}
		//std::cout << "current size of inliers: " << tmp_inliers.size() << ", so far best size: " << inliers_.size() << std::endl;

		// If the current sphere has more inliers than the previous so-far-the-best, update the best parameters
		if (tmp_inliers.size() > inliers_.size())
		{
			tmp_inliers.swap(inliers_);
			sphere_params_.at<double>(0) = cx;
			sphere_params_.at<double>(1) = cy;
			sphere_params_.at<double>(2) = cz;
			sphere_params_.at<double>(3) = r;
			std::cout << "Iteration number before update: " << iteration_number << ", ";
			iteration_number = getIterationNumber(confidence, inliers_.size(), points_->size(), 4);
			std::cout << "Iteration number after update: " << iteration_number << std::endl;
		}

	}//----end of iterations loop

	std::cout << "Number of iterations to do has been reached." << std::endl;
	// Clean up the memory
	delete[] sample;
}

void removeInliers(std::vector<cv::Point3d>& points_, std::vector<int>& inliers_)
{
	std::vector<cv::Point3d> temp;
	for (size_t i = 0; i < points_.size(); ++i)
	{
		if (std::find(inliers_.begin(), inliers_.end(), i) == inliers_.end())
		{
			temp.push_back(points_[i]);
		}
	}
	points_.swap(temp);
}

void RansacSphereDriver(double th, double conf)
{
	// set a random seed
	unsigned int time_ui = unsigned int(time(NULL));
	srand(time_ui);

	std::cout << "reading in: C:/EreBere/Project/ELTE/grafika/res/Sphere1234togeather.txt\n";
	std::ifstream points_file("C:/EreBere/Project/ELTE/grafika/res/Sphere1234togeather.txt");
	std::string line;
	const std::string delimiter = " ";
	std::vector<cv::Point3d> points;
	double coord1, coord2, coord3;
	while (!points_file.eof())
	{
		std::getline(points_file, line);

		int next_space = line.find(delimiter);
		coord1 = std::stod(line.substr(0, next_space).c_str());
		line = line.substr(next_space + 1);

		next_space = line.find(delimiter);
		coord2 = std::stod(line.substr(0, next_space).c_str());
		line = line.substr(next_space + 1);

		coord3 = std::stod(line.c_str());

		points.push_back(cv::Point3d(coord1, coord2, coord3));
	}
	int ere = 0;

	bool go_on = true;
	while (go_on)
	{
		// run RANSAC fitting
		std::vector<int> inliers;
		cv::Mat sphere_params;
		fitSphereRANSAC(&points, inliers, sphere_params, th, conf);

		// print the results
		std::cout << "=====================================FOUND A SPHERE=================================\n";
		std::cout << "\n\tnumber of inliers: " << inliers.size() << "\n";
		std::cout << "\tcenter coordinates: [" << sphere_params.at<double>(0) << ", " << sphere_params.at<double>(1) << ", " << sphere_params.at<double>(2) << "] radius = " << sphere_params.at<double>(3) << "\n\n";
		std::cout << "=====================================================================================\n\n";

		// remove inliers from points
		removeInliers(points, inliers);
		std::cout << "new number of points = " << points.size() << "\n";

		// check if it makes sense to continue
		go_on = inliers.size() > 150;
	}

	return;
}