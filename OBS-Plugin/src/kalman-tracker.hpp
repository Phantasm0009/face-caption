#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

struct KalmanTracker {
	cv::KalmanFilter kf;
	bool initialized = false;
	float vx = 0.f, vy = 0.f; // velocity for lookahead

	KalmanTracker();
	void update(float cx, float ty, int face_w, int face_h);
	void predict(float *out_cx, float *out_ty);
	void get_velocity(float *out_vx, float *out_vy) const;
};
