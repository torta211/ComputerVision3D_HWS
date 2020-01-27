#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"

void drawPoints(const std::vector<cv::Point2d>& points_, cv::Mat& image_, const cv::Scalar& color_, const double& size_);

void drawInliers(const std::vector<cv::Point2d>& points_, cv::Mat& image_, const cv::Scalar& color_, const std::vector<int> inliers_, const double& size_);

size_t getIterationNumber(double confidence_, size_t inlier_number_, size_t point_number_, size_t sample_size_);

void fitCircleRANSAC(const std::vector<cv::Point2d>* const points_, std::vector<int>& inliers_, cv::Mat& circle_params_, double threshold, double confidence, cv::Mat& canvas);

cv::Mat position_image(const cv::Mat& original_, int dx, int dy);

void RansacCircleDriver(double th, double conf, int config);