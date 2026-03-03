#pragma once

#include <obs-module.h>
#include <string>
#include <mutex>
#include <atomic>
#include <memory>
#include <thread>

struct FaceDetector;
struct KalmanTracker;
struct CaptionRenderer;
struct STTVosk;

struct face_captions_filter_data {
	obs_source_t *context = nullptr;

	std::unique_ptr<FaceDetector> face_detector;
	std::unique_ptr<KalmanTracker> kalman;
	std::unique_ptr<CaptionRenderer> caption_renderer;

	float face_center_x = 0.0f;
	float face_top_y = 0.0f;
	int face_bbox[4] = {0, 0, 0, 0};
	bool face_detected = false;
	float smooth_capt_x = 0.f, smooth_capt_y = 0.f;
	bool smooth_capt_valid = false;

	std::mutex text_mutex;
	std::string caption_text;
	std::string caption_partial;
	std::string caption_history[2];
	double last_caption_time = 0.0;
	static constexpr double CAPTION_TIMEOUT = 4.5;

	uint32_t base_width = 1280;
	uint32_t base_height = 720;
	int font_size = 52;
	int caption_max_width = 620;
	int caption_offset_y = 0;
	bool show_face_box = false;
	bool speech_bubble_tail = true;
	int caption_history_lines = 2;
	std::string chroma_color = "green";
	bool chroma_key_mode = false;
	std::string vosk_model_path;

	std::unique_ptr<STTVosk> stt;
	std::atomic<bool> stt_ready{false};
	std::thread model_load_thread;
	std::atomic<bool> model_load_abort{false};
	int obs_audio_rate = 48000;
	int input_audio_rate = 44100;  // source output rate (44100 or 48000) for resample to 16kHz

	std::atomic<bool> active{false};
	int frame_count = 0;
};

extern obs_source_info face_captions_filter_info;
