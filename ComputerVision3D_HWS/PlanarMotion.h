#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
#include "CircleFittingRansac.h"
#include "HomographyNormalized.h"

//void normalizeCoords(std::vector<cv::Point2f>&, std::vector<cv::Point2f>&, cv::Mat&, cv::Mat&);

void planarMotionDriver();

cv::Mat fundamentalRANSAC(const std::vector<cv::Point2f>&, const std::vector<cv::Point2f>& );

// size_t getIterationNumber(double, size_t, size_t, size_t);
