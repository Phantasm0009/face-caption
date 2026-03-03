#include "face-detector.hpp"
#include <obs-module.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

FaceDetector::FaceDetector() { load_cascade(); }

bool FaceDetector::load_cascade(const char *path)
{
	if (path && path[0] && cascade.load(path)) { cascade_path = path; return true; }
	char *mod_path = obs_module_file("haarcascade_frontalface_default.xml");
	if (mod_path && cascade.load(mod_path)) { cascade_path = mod_path; bfree(mod_path); return true; }
	bfree(mod_path);
#ifdef _WIN32
	const char *candidates[] = { "haarcascade_frontalface_default.xml",
		"C:/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
		"C:/vcpkg/installed/x64-windows/share/opencv4/haarcascades/haarcascade_frontalface_default.xml", nullptr };
#else
	const char *candidates[] = { "/usr/share/opencv4/data/haarcascades/haarcascade_frontalface_default.xml",
		"/usr/share/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
		"haarcascade_frontalface_default.xml", nullptr };
#endif
	for (int i = 0; candidates[i]; i++)
		if (cascade.load(candidates[i])) { cascade_path = candidates[i]; return true; }
	return false;
}

cv::Rect FaceDetector::detect(const cv::Mat &frame, float prefer_cx, float prefer_ty, bool currently_tracking)
{
	if (frame.empty() || cascade.empty()) return cv::Rect();

	// Skip frames for performance (like Python version)
	int interval = currently_tracking ? detect_interval_tracking : detect_interval_lost;
	frame_counter++;
	if (frame_counter % interval != 0 && currently_tracking && last_face.width > 0) {
		return last_face; // reuse last detection
	}

	// Downscale for speed (like Python: 0.20-0.40 scale)
	float scale = currently_tracking ? detect_scale_tracking : detect_scale_lost;
	cv::Mat gray;
	if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
	else if (frame.channels() == 4) cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
	else gray = frame;

	cv::Mat small_gray;
	int sw = (int)(gray.cols * scale);
	int sh = (int)(gray.rows * scale);
	if (sw < 60 || sh < 60) { sw = gray.cols; sh = gray.rows; scale = 1.0f; }
	cv::resize(gray, small_gray, cv::Size(sw, sh));

	std::vector<cv::Rect> faces;
	int min_sz = (int)(min_face_size * scale);
	if (min_sz < 15) min_sz = 15;
	cascade.detectMultiScale(small_gray, faces, 1.1, 4, 0, cv::Size(min_sz, min_sz));
	if (faces.empty()) {
		last_face = cv::Rect();
		return cv::Rect();
	}

	// Scale back to original coordinates
	float inv_scale = 1.0f / scale;
	for (auto &f : faces) {
		f.x = (int)(f.x * inv_scale);
		f.y = (int)(f.y * inv_scale);
		f.width = (int)(f.width * inv_scale);
		f.height = (int)(f.height * inv_scale);
	}

	// Pick face closest to previous position (like Python)
	int best = 0;
	float best_score = 1e9f;
	for (size_t i = 0; i < faces.size(); i++) {
		float cx = faces[i].x + faces[i].width * 0.5f;
		float ty = (float)faces[i].y;
		float d = std::abs(cx - prefer_cx) + std::abs(ty - prefer_ty) * 0.5f;
		if (d < best_score) { best_score = d; best = (int)i; }
	}
	last_face = faces[best];
	return last_face;
}
