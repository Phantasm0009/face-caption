#include "kalman-tracker.hpp"

KalmanTracker::KalmanTracker()
	: kf(4, 2, 0, CV_32F)
{
	// State: [cx, ty, vx, vy]  Measurement: [cx, ty]
	cv::setIdentity(kf.transitionMatrix);
	kf.transitionMatrix.at<float>(0, 2) = 1.f; // cx += vx
	kf.transitionMatrix.at<float>(1, 3) = 1.f; // ty += vy
	cv::setIdentity(kf.measurementMatrix);
	// Match Python parameters for smooth tracking
	cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(0.003f));
	cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(0.25f));
	cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
}

void KalmanTracker::update(float cx, float ty, int face_w, int face_h)
{
	(void)face_w;
	(void)face_h;
	cv::Mat measure = (cv::Mat_<float>(2, 1) << cx, ty);
	if (!initialized) {
		kf.statePost.at<float>(0, 0) = cx;
		kf.statePost.at<float>(1, 0) = ty;
		kf.statePost.at<float>(2, 0) = 0.f;
		kf.statePost.at<float>(3, 0) = 0.f;
		initialized = true;
	}
	kf.predict(); // predict before correct for proper Kalman cycle
	kf.correct(measure);
	// Store velocity for lookahead
	vx = kf.statePost.at<float>(2, 0);
	vy = kf.statePost.at<float>(3, 0);
}

void KalmanTracker::predict(float *out_cx, float *out_ty)
{
	if (!initialized) {
		if (out_cx) *out_cx = 0.f;
		if (out_ty) *out_ty = 0.f;
		return;
	}
	cv::Mat pred = kf.predict();
	if (out_cx) *out_cx = pred.at<float>(0, 0);
	if (out_ty) *out_ty = pred.at<float>(1, 0);
	vx = pred.at<float>(2, 0);
	vy = pred.at<float>(3, 0);
}

void KalmanTracker::get_velocity(float *out_vx, float *out_vy) const
{
	if (out_vx) *out_vx = vx;
	if (out_vy) *out_vy = vy;
}
