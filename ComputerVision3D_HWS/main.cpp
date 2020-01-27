#include <iostream>
#include <io.h>
#include <fcntl.h>
#include "CircleFittingRansac.h"
#include "SphereFittingRansac.h"
#include "HomographyNormalized.h"
#include "Pointclouds.h"
#include "PlanarMotion.h"

void chooseprogram();

int wmain()
{
	// to be able to display special characters
	int prev = _setmode(_fileno(stdout), _O_U16TEXT);
	std::wcout << L"3D computer vision homework compilation. \n \u00a9 Kotán Tamás 2019/2020 @ ELTE IK Computer Graphics\n";
	// switch back, so we avoid assertion failures with cin
	_setmode(_fileno(stdout), prev);

	chooseprogram();

	return 1;
}

void chooseprogram()
{
	std::cout << "\nType in a number to run a program.\n";
	std::cout << "\t(0) - EXIT\n";
	std::cout << "\t(1) - circle fitting with RANSAC\n";
	std::cout << "\t(2) - sphere fitting with RANSAC\n";
	std::cout << "\t(3) - normalized linear homography estimation\n";
	std::cout << "\t(4) - Planar stereo view\n";
	std::cout << "\t(5) - ICP for lidar pointclouds\n";

	int choice;
	std::cin >> choice;
	
	switch (choice)
	{
	case 0:
	{
		std::cout << "OK. EXITING." << std::endl;
		return;
	}
	case 1:
	{
		double th = 1.5;
		double conf = 0.999;
		int config = 2;
		std::cout << "Running: Homework No. 1. with threshold/confidence/configuration: " << th << "/" << conf << "/" << config << "\n";
		RansacCircleDriver(th, conf, config);
		break;
	}
	case 2:
	{
		double th = 0.01;
		double conf = 0.5;
		std::cout << "Running: Homework No. 2. with threshold/confidence: " << th << "/" << conf << "\n";
		RansacSphereDriver(th, conf);
		break;
	}
	case 3:
	{
		int config = 1;
		bool swap = true;
		std::cout << "Running: Homework No. 3. with config: " << config << std::string(swap ? " (swapped)" : "") << "\n";
		HomographyDriver(config, swap);
		break;
	}
	case 4:
	{
		std::cout << "Running: Homework No. 4.\n";
		planarMotionDriver();
		break;
	}
	case 5:
	{
		int iters = 2;
		std::cout << "Running: Homework No. 5. with " << iters << " iterations\n";
		pointcloudsDriver(iters);
		break;
	}
	default:
	{
		std::cout << "\nThat was not a valid choice.\n\n";
		break;
	}
	}
	chooseprogram();
}