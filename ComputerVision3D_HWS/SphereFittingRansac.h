#pragma once

#include "CircleFittingRansac.h"
void fitSphereRANSAC(const std::vector<cv::Point3d>* const points_, std::vector<int>& inliers_, cv::Mat& sphere_params_, double threshold, double confidence);

void removeInliers(std::vector<cv::Point3d>& points_, std::vector<int>& inliers_);

void RansacSphereDriver(double th, double conf);