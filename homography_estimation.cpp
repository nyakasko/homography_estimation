#include <iostream>
#include <stdio.h>  
#include <opencv2/opencv.hpp>  
#include <vector>
#include <string> 
#include <fstream>
#include <sstream>
#include <omp.h>

using namespace cv;
using namespace std;
#define SQRT2 1.41

void detectFeatures(
	const Mat& image1_,
	const Mat& image2_,
	vector<Point2f>& source_points_,
	vector<Point2f>& destination_points_)
{
	printf("Detect SIFT features\n");
	Mat descriptors1, descriptors2; // The descriptors of the found keypoints in the two images
	vector<KeyPoint> keypoints1, keypoints2; // The keypoints in the two images

	Ptr<SIFT> detector = SIFT::create(); // The SIFT detector
	detector->detect(image1_, keypoints1); // Detecting keypoints in the first image
	detector->compute(image1_, keypoints1, descriptors1); // Computing the descriptors of the keypoints in the first image
	printf("Features found in the first image: %d\n", keypoints1.size());

	detector->detect(image2_, keypoints2); // Detecting keypoints in the second image
	detector->compute(image2_, keypoints2, descriptors2); // Computing the descriptors of the keypoints in the second image
	printf("Features found in the second image: %d\n", keypoints2.size());

	// Do the descriptor matching by an approximated k-nearest-neighbors algorithm (FLANN) with k = 2.
	vector<vector<DMatch>> matches_vector;
	FlannBasedMatcher matcher(new flann::KDTreeIndexParams(5), new flann::SearchParams(32));
	matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2); // K-nearest-neighbors

	// Iterate through the matches, apply the SIFT ratio test and if the match passes,
	// add it to the vector of found correspondences
	for (auto m : matches_vector)
	{
		if (m.size() == 2 && m[0].distance < m[1].distance * 0.80)
		{
			auto& kp1 = keypoints1[m[0].queryIdx];
			auto& kp2 = keypoints2[m[0].trainIdx];
			source_points_.push_back(kp1.pt);
			destination_points_.push_back(kp2.pt);
		}
	}
	printf("Detected correspondence number: %d\n", source_points_.size());
}

void normalizePoints(
	const vector<Point2f>& input_source_points_,
	const vector<Point2f>& input_destination_points_,
	vector<Point2f>& output_source_points_,
	vector<Point2f>& output_destination_points_,
	Mat& T1_,
	Mat& T2_)
{
	T1_ = Mat::eye(3, 3, CV_32F);
	T2_ = Mat::eye(3, 3, CV_32F);

	const size_t pointNumber = input_source_points_.size();
	output_source_points_.resize(pointNumber);
	output_destination_points_.resize(pointNumber);
	float mean_source_x = 0.0, mean_source_y = 0.0, mean_destination_x = 0.0, mean_destination_y = 0.0;

	for (auto i = 0; i < pointNumber; ++i)
	{
		mean_source_x += input_source_points_[i].x;
		mean_source_y += input_source_points_[i].y;
		mean_destination_x += input_destination_points_[i].x;
		mean_destination_y += input_destination_points_[i].y;
	}
	mean_source_x /= pointNumber;
	mean_source_y /= pointNumber;
	mean_destination_x /= pointNumber;
	mean_destination_y /= pointNumber;

	float spread_source_x = 0.0, spread_source_y = 0.0, spread_destination_x = 0.0, spread_destination_y = 0.0;
	for (auto i = 0; i < pointNumber; ++i) {
		spread_source_x += (input_source_points_[i].x - mean_source_x) * (input_source_points_[i].x - mean_source_x);
		spread_source_y += (input_source_points_[i].y - mean_source_y) * (input_source_points_[i].y - mean_source_y);
		spread_destination_x += (input_destination_points_[i].x - mean_destination_x) * (input_destination_points_[i].x - mean_destination_x);
		spread_destination_y += (input_destination_points_[i].y - mean_destination_y) * (input_destination_points_[i].y - mean_destination_y);
	}

	spread_source_x /= pointNumber;
	spread_source_y /= pointNumber;
	spread_destination_x /= pointNumber;
	spread_destination_y /= pointNumber;

	Mat offs_source = Mat::eye(3, 3, CV_32F);
	Mat scale_source = Mat::eye(3, 3, CV_32F);
	Mat offs_destination = Mat::eye(3, 3, CV_32F);
	Mat scale_destination = Mat::eye(3, 3, CV_32F);

	offs_source.at<float>(0, 2) = -mean_source_x;
	offs_source.at<float>(1, 2) = -mean_source_y;
	offs_destination.at<float>(0, 2) = -mean_destination_x;
	offs_destination.at<float>(1, 2) = -mean_destination_y;

	scale_source.at<float>(0, 0) = SQRT2 / sqrt(spread_source_x);
	scale_source.at<float>(1, 1) = SQRT2 / sqrt(spread_source_y);
	scale_destination.at<float>(0, 0) = SQRT2 / sqrt(spread_destination_x);
	scale_destination.at<float>(1, 1) = SQRT2 / sqrt(spread_destination_y);

	T1_ = scale_source * offs_source;
	T2_ = scale_destination * offs_destination;

	for (auto i = 0; i < pointNumber; ++i)
	{
		Point2f p2D;

		p2D.x = SQRT2 * (input_source_points_[i].x - mean_source_x) / sqrt(spread_source_x);
		p2D.y = SQRT2 * (input_source_points_[i].y - mean_source_y) / sqrt(spread_source_y);
		output_source_points_[i] = p2D;
		p2D.x = SQRT2 * (input_destination_points_[i].x - mean_destination_x) / sqrt(spread_destination_x);
		p2D.y = SQRT2 * (input_destination_points_[i].y - mean_destination_y) / sqrt(spread_destination_y);
		output_destination_points_[i] = p2D;
	}
}

Mat calcHomography(
	vector<pair<Point2f, Point2f>>
	pointPairs)
{
	const int ptsNum = pointPairs.size();
	Mat A(2 * ptsNum, 9, CV_32F);
	for (auto i = 0; i < ptsNum; i++) {
		float u1 = pointPairs[i].first.x;
		float v1 = pointPairs[i].first.y;

		float u2 = pointPairs[i].second.x;
		float v2 = pointPairs[i].second.y;

		A.at<float>(2 * i, 0) = u1;
		A.at<float>(2 * i, 1) = v1;
		A.at<float>(2 * i, 2) = 1.0f;
		A.at<float>(2 * i, 3) = 0.0f;
		A.at<float>(2 * i, 4) = 0.0f;
		A.at<float>(2 * i, 5) = 0.0f;
		A.at<float>(2 * i, 6) = -u2 * u1;
		A.at<float>(2 * i, 7) = -u2 * v1;
		A.at<float>(2 * i, 8) = -u2;

		A.at<float>(2 * i + 1, 0) = 0.0f;
		A.at<float>(2 * i + 1, 1) = 0.0f;
		A.at<float>(2 * i + 1, 2) = 0.0f;
		A.at<float>(2 * i + 1, 3) = u1;
		A.at<float>(2 * i + 1, 4) = v1;
		A.at<float>(2 * i + 1, 5) = 1.0f;
		A.at<float>(2 * i + 1, 6) = -v2 * u1;
		A.at<float>(2 * i + 1, 7) = -v2 * v1;
		A.at<float>(2 * i + 1, 8) = -v2;

	}

	Mat eVecs(9, 9, CV_32F), eVals(9, 9, CV_32F);
	eigen(A.t() * A, eVals, eVecs);

	Mat H(3, 3, CV_32F);
	for (int i = 0; i < 9; i++) H.at<float>(i / 3, i % 3) = eVecs.at<float>(8, i);

	return H;
}

void transformImage(
	Mat origImg,
	Mat& newImage,
	Mat tr,
	bool isPerspective)
{
	Mat invTr = tr.inv();
	const int WIDTH = origImg.cols;
	const int HEIGHT = origImg.rows;
	const int newWIDTH = newImage.cols;
	const int newHEIGHT = newImage.rows;

	for (int x = 0; x < newWIDTH; x++) for (int y = 0; y < newHEIGHT; y++) {
		Mat pt(3, 1, CV_32F);
		pt.at<float>(0, 0) = x;
		pt.at<float>(1, 0) = y;
		pt.at<float>(2, 0) = 1.0;

		Mat ptTransformed = invTr * pt;
		if (isPerspective) ptTransformed = (1.0 / ptTransformed.at<float>(2, 0)) * ptTransformed;

		int newX = round(ptTransformed.at<float>(0, 0));
		int newY = round(ptTransformed.at<float>(1, 0));

		if ((newX >= 0) && (newX < WIDTH) && (newY >= 0) && (newY < HEIGHT)) newImage.at<Vec3b>(y, x) = origImg.at<Vec3b>(newY, newX);
	}
}

int getIterationNumber(int point_number_,
	int inlier_number_,
	int sample_size_,
	double confidence_)
{
	const double inlier_ratio = static_cast<float>(inlier_number_) / point_number_;

	static const double log1 = log(1.0 - confidence_);
	const double log2 = log(1.0 - pow(inlier_ratio, sample_size_));

	const int k = log1 / log2;
	if (k < 0)
		return INT_MAX;
	return k;
}

Mat ransacHMatrix(
	const vector<Point2f>& normalized_input_src_points_,
	const vector<Point2f>& normalized_input_destination_points_,
	const Mat& T1_,
	const Mat& T2_)
{
	srand(time(NULL));
	// The so-far-the-best H
	Mat best_H(3, 3, CV_32F);
	// The number of correspondences
	const size_t point_number = normalized_input_src_points_.size();
	float prev_error = numeric_limits<double>::max();
	// Initializing the index pool from which the minimal samples are selected
	vector<size_t> index_pool(point_number);
	for (size_t i = 0; i < point_number; ++i)
		index_pool[i] = i;

	// The size of a minimal sample
	constexpr size_t sample_size = 8;
	// The minimal sample
	size_t* mss = new size_t[sample_size];

	size_t iteration_limit = 100000, // A strict iteration limit which mustn't be exceeded
		iteration = 0; // The current iteration number

	vector<Point2f> source_points(sample_size),
		destination_points(sample_size);

	while (iteration++ < iteration_limit)
	{
		vector<pair<Point2f, Point2f>> pointPairs;
		for (auto sample_idx = 0; sample_idx < sample_size; ++sample_idx)
		{
			// Select a random index from the pool
			const size_t idx = round((rand() / (double)RAND_MAX) * (index_pool.size() - 1));
			mss[sample_idx] = index_pool[idx];
			index_pool.erase(index_pool.begin() + idx);

			// Put the selected correspondences into the point containers
			const size_t point_idx = mss[sample_idx];
			source_points[sample_idx] = normalized_input_src_points_[point_idx];
			destination_points[sample_idx] = normalized_input_destination_points_[point_idx];

			pair<Point2f, Point2f> Point;
			Point.first = source_points[sample_idx];
			Point.second = destination_points[sample_idx];
			pointPairs.push_back(Point);
		}

		// Estimate H matrix
		Mat H_(3, 3, CV_32F);

		H_ = calcHomography(pointPairs);

		int errorCalcIdNum = 100;
		vector<Point2f> errorCalc_source_points(errorCalcIdNum),
			errorCalc_destination_points(errorCalcIdNum);

		double error1 = 0.0;
		double error2 = 0.0;
		for (auto i = 0; i < errorCalcIdNum; ++i) {
			const size_t randID = round((rand() / (double)RAND_MAX) * (normalized_input_src_points_.size() - 1));
			errorCalc_source_points[i] = normalized_input_src_points_[randID];
			errorCalc_destination_points[i] = normalized_input_destination_points_[randID];
			Mat pt1(3, 1, CV_32F);
			pt1.at<float>(0, 0) = errorCalc_source_points[i].x; 
			pt1.at<float>(1, 0) = errorCalc_source_points[i].y;
			pt1.at<float>(2, 0) = 1;
			Mat pt2(3, 1, CV_32F);
			pt2.at<float>(0, 0) = errorCalc_destination_points[i].x;
			pt2.at<float>(1, 0) = errorCalc_destination_points[i].y;
			pt2.at<float>(2, 0) = 1;
			error1 += pow(norm(H_ * pt1, pt2),2);
			error2 += pow(norm(pt1, H_.inv() * pt2),2);
		}
		double average_error = (double)((error1 + error2) / point_number);
		// Update if the new model is better than the previous so-far-the-best.
		if (average_error < prev_error)
		{
			prev_error = average_error;
			best_H = H_;
			cout << average_error << endl;
		}

		// Put back the selected points to the pool
		for (size_t i = 0; i < sample_size; ++i)
			index_pool.push_back(mss[i]);
	}
	return best_H;
}

int main(int argc, char* argv[])
{
	srand(time(NULL));
	// Load images
	Mat image1 = imread("D:/source/repos/homography_estimation/data/horvat2.png");
	Mat image2 = imread("D:/source/repos/homography_estimation/data/horvat1.png");

	// Detect features
	vector<Point2f> source_points, destination_points; // Point correspondences
	detectFeatures(image1, // First image
		image2, // Second image
		source_points, // Points in the first image 
		destination_points); // Points in the second image 

	// Normalize the coordinates of the point correspondences to achieve numerically stable results
	Mat T1(3, 3, CV_32F), T2(3, 3, CV_32F); // Normalizing transformations
	vector<Point2f> normalized_source_points, normalized_destination_points; // Normalized point correspondences
	normalizePoints(source_points, // Points in the first image 
		destination_points,  // Points in the second image
		normalized_source_points,  // Normalized points in the first image
		normalized_destination_points, // Normalized points in the second image
		T1, // Normalizing transformation in the first image
		T2); // Normalizing transformation in the second image

	Mat best_H(3, 3, CV_32F);
	best_H = ransacHMatrix(
		normalized_source_points,  // Normalized points in the first image 
		normalized_destination_points, // Normalized points in the second image
		T1, // Normalizing transforcv::Mation in the first image
		T2); // Normalizing transforcv::Mation in the second image

	best_H = T2.inv() * best_H * T1; // Denormalize the H matrix
	best_H = best_H * (1.0 / best_H.at<float>(2, 2));
	cout << best_H << endl;

	Mat transformedImage = Mat::zeros(image2.size().height,image2.size().width, image2.type());
	//transformImage(image2, transformedImage, Mat::eye(3, 3, CV_32F), true);
	transformImage(image1, transformedImage, best_H, true);
	namedWindow("Display frame", WINDOW_NORMAL);
	imshow("Display frame", transformedImage);
	waitKey();
	imwrite("D:/source/repos/homography_estimation/data/res.png", transformedImage);

	return 0;
}
