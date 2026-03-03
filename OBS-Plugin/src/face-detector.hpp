#pragma once

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <string>

struct FaceDetector {
	cv::CascadeClassifier cascade;
	std::string cascade_path;

	// Detection frequency: skip frames for performance (like Python version)
	int detect_interval_tracking = 6;  // every 6th frame when tracking
	int detect_interval_lost = 1;      // every frame when face lost
	float detect_scale_tracking = 0.25f; // downscale when tracking (fast)
	float detect_scale_lost = 0.40f;     // downscale when lost (more coverage)
	int min_face_size = 30;

	// State
	int frame_counter = 0;
	bool was_tracking = false;
	cv::Rect last_face;

	FaceDetector();
	bool load_cascade(const char *path = nullptr);
	cv::Rect detect(const cv::Mat &frame, float prefer_cx, float prefer_ty, bool currently_tracking);
};
