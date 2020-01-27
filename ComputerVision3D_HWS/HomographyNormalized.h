#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"

void normalizeCoords(std::vector<cv::Point2f>&, std::vector<cv::Point2f>&, cv::Mat&, cv::Mat&);

cv::Mat estHomographyMatrix(std::vector<cv::Point2f>, std::vector<cv::Point2f>);

void transformImage(cv::Mat, cv::Mat&, cv::Mat, bool);

void HomographyDriver(int, bool);