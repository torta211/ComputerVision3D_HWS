#include "Pointclouds.h"

// this could be used in other homeworks too...
std::vector<cv::Point3f> readXYZFile(std::string filename)
{
	std::cout << "reading in: " << filename << "\n";

	std::ifstream points_file(filename);
	std::vector<cv::Point3f> pc;

	std::string line;
	double x, y, z;
	int next_space;
	const std::string delimiter = " ";

	while (!points_file.eof())
	{
		std::getline(points_file, line);

		next_space = line.find(delimiter);
		x = std::stod(line.substr(0, next_space).c_str());
		line = line.substr(next_space + 1);

		next_space = line.find(delimiter);
		y = std::stod(line.substr(0, next_space).c_str());
		line = line.substr(next_space + 1);

		next_space = line.find(delimiter);
		z = std::stod(line.substr(0, next_space).c_str());

		pc.push_back(cv::Point3f(x, y, z));
	}

	return pc;
}

// euclidean distance for two cv::Point3f
double eucDist3D(cv::Point3f a, cv::Point3f b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

double distanceOfMatches(const std::vector<cv::Point3f>& target, const std::vector<cv::Point3f>& tomove, const std::vector<int>& matches)
{
	double dist = 0;
	for (int i = 0; i < tomove.size(); i++)
	{
		dist += eucDist3D(tomove[i], target[matches[i]]);
	}
	return dist;
}

// this function finds the index of closest point in points_target for every point of points_tomove
std::vector<int> findMatches(const std::vector<cv::Point3f>& target, const std::vector<cv::Point3f>& tomove)
{
	std::vector<int> matches;
	int match_index;
	double min_distance;
	double current_distance;
	for (int i = 0; i < tomove.size(); i++)
	{
		match_index = 0;
		cv::Point3f point = tomove[i];
		min_distance = std::numeric_limits<double>::max();
		for (int j = 0; j < target.size(); j++)
		{
			current_distance = eucDist3D(point, target[j]);
			if (current_distance < min_distance)
			{
				min_distance = current_distance;
				match_index = j;
			}
		}
		if (i % 500 == 0) std::cout << "match: " << i << " and " << match_index << "\n";
		matches.push_back(match_index);
	}
	return matches;
}

// transforms pointcloud tomove, so that the distance of the matches is minimized
void fitMatches(const std::vector<cv::Point3f>& target, std::vector<cv::Point3f>& tomove, const std::vector<int>& matches, bool doScaling)
{
	// calculate vector from centor of gravity of tomove to center of gravity of target
	cv::Point3f center_target(0.0f, 0.0f, 0.0f);
	cv::Point3f center_tomove(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < tomove.size(); i++)
	{
		center_tomove += tomove[i];
		center_target += target[matches[i]];
	}
	center_tomove /= int(tomove.size());
	center_target /= int(tomove.size());	
	cv::Point3f translation_vector = center_target - center_tomove;

	// we apply this first
	for (int i = 0; i < tomove.size(); tomove[i++] += translation_vector);

	// find a matrix that rotates each point near its target match
	cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);
	for (int i = 0; i < tomove.size(); i++)
	{
		cv::Point3f tomove_pt = tomove[i];
		cv::Point3f target_pt = target[matches[i]];
		float x1 = target_pt.x;
		float y1 = target_pt.y;
		float z1 = target_pt.z;
		float x2 = tomove_pt.x;
		float y2 = tomove_pt.y;
		float z2 = tomove_pt.z;

		H.at<float>(0, 0) += x2 * x1;
		H.at<float>(0, 1) += x2 * y1;
		H.at<float>(0, 2) += x2 * z1;
		H.at<float>(1, 0) += y2 * x1;
		H.at<float>(1, 1) += y2 * y1;
		H.at<float>(1, 2) += y2 * z1;
		H.at<float>(2, 0) += z2 * x1;
		H.at<float>(2, 1) += z2 * y1;
		H.at<float>(2, 2) += z2 * z1;
	}

	cv::Mat W(3, 3, CV_32F);
	cv::Mat U(3, 3, CV_32F);
	cv::Mat Vt(3, 3, CV_32F);
	cv::SVD::compute(H, W, U, Vt);
	cv::Mat ROT = Vt.t() * U.t();

	// apply rotation
	for (int i = 0; i < tomove.size(); i++)
	{
		cv::Mat point(3, 1, CV_32F);
		point.at<float>(0, 0) = tomove[i].x;
		point.at<float>(1, 0) = tomove[i].y;
		point.at<float>(2, 0) = tomove[i].z;

		point = ROT * point;

		tomove[i] = cv::Point3f(point.at<float>(0, 0), point.at<float>(1, 0), point.at<float>(2, 0));
	}

	// if we specified scaling as required, we scale
	if (doScaling)
	{
		// TODO: scaling
	}
}

void pointcloudsDriver(int iterations)
{
	// points_target will be fixed and we try to move points_tomove to overlap it best
	std::cout << "Target pointcloud:\n";
	std::vector<cv::Point3f> points_target = readXYZFile("C:/EreBere/Project/ELTE/grafika/res/pc3.xyz");
	std::cout << "Pointcloud to transform:\n";
	std::vector<cv::Point3f> points_tomove = readXYZFile("C:/EreBere/Project/ELTE/grafika/res/pc4.xyz");

	// since we are dealing with lidar data of the same scene, but the lidar moved, it could be a good idea, to first translate the second pointclouds center to the first one's
	// calculate vector from center of gravity of tomove, to center of gravity of target
	cv::Point3f center_target(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < points_target.size(); i++)
	{
		center_target += points_target[i];
	}
	center_target /= int(points_target.size());
	cv::Point3f center_tomove(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < points_tomove.size(); i++)
	{
		center_tomove += points_tomove[i];
	}
	center_tomove /= int(points_tomove.size());

	cv::Point3f translation_vector = center_target - center_tomove;
	for (int i = 0; i < points_tomove.size(); points_tomove[i++] += translation_vector);

	// start ICP
	int iterations_done = 0;
	while (iterations_done < iterations)
	{
		std::vector<int> matches = findMatches(points_target, points_tomove);

		std::cout << "error before transformation = " << distanceOfMatches(points_target, points_tomove, matches) << "\n";

		fitMatches(points_target, points_tomove, matches, false); // no scaling for now

		std::cout << "error after transformation = " << distanceOfMatches(points_target, points_tomove, matches) << "\n";

		iterations_done += 1;
		std::cout << iterations_done << ". iteration done.\n";
	}
	std::cout << "\n DONE. Do You want to write the transformed pointcloud into a file? (y/n)\n";

	std::string choice;
	std::cin >> choice;
	if (choice == "y")
	{
		std::ofstream file_out;
		file_out.open("C:/EreBere/Project/ELTE/grafika/res/transformedPointcloud_mod.xyz");
		for (int i = 0; i < points_tomove.size(); i++)
		{
			file_out << points_tomove[i].x << " " << points_tomove[i].y << " " << points_tomove[i].z << "\n";
		}
		file_out.close();
		std::cout << "File generated.\n";
	}
	return;
}