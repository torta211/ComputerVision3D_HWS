#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"

std::vector<cv::Point3f> readXYZFile(std::string);

double eucDist3D(cv::Point3f, cv::Point3f);

double distanceOfMatches(const std::vector<cv::Point3f>&, const std::vector<cv::Point3f>&, const std::vector<int>&);

std::vector<int> findMatches(const std::vector<cv::Point3f>&, const std::vector<cv::Point3f>&);

void fitMatches(const std::vector<cv::Point3f>&, std::vector<cv::Point3f>&, const std::vector<int>&, bool);

void pointcloudsDriver(int);