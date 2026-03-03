#include "face-captions-filter.hpp"
#include "face-detector.hpp"
#include "caption-renderer.hpp"
#include "kalman-tracker.hpp"
#include "stt-vosk.hpp"
#include <obs-module.h>
#include <obs.h>
#include <obs-defs.h>
#include <media-io/video-io.h>
#include <util/platform.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <chrono>

#define S_FONT_SIZE "caption_font_size"
#define S_MAX_WIDTH "caption_max_width"
#define S_OFFSET_Y "caption_offset_y"
#define S_SHOW_FACE_BOX "show_face_box"
#define S_SPEECH_BUBBLE "speech_bubble_tail"
#define S_HISTORY_LINES "caption_history_lines"
#define S_CHROMA_MODE "chroma_key_mode"
#define S_CHROMA_COLOR "chroma_color"
#define S_VOSK_MODEL "vosk_model_path"
#define S_AUDIO_RATE "audio_input_rate"

#define BUNDLED_MODEL_SUBDIR "models/vosk-model-small-en-us-0.15"
#define LARGE_MODEL_SUBDIR   "models/vosk-model-en-us-0.22"

static char *get_best_vosk_model_path(void)
{
	char *large_cfg = obs_module_file(LARGE_MODEL_SUBDIR "/conf/model.conf");
	if (large_cfg) {
		char *slash = strstr(large_cfg, "/conf");
#ifdef _WIN32
		if (!slash) slash = strstr(large_cfg, "\\conf");
#endif
		if (slash) { *slash = '\0'; return large_cfg; }
		bfree(large_cfg);
	}
	char *cfg = obs_module_file(BUNDLED_MODEL_SUBDIR "/conf/model.conf");
	if (!cfg) return nullptr;
	char *slash = strstr(cfg, "/conf");
#ifdef _WIN32
	if (!slash) slash = strstr(cfg, "\\conf");
#endif
	if (slash) *slash = '\0';
	return cfg;
}

static char *get_bundled_model_path(void)
{
	char *cfg = obs_module_file(BUNDLED_MODEL_SUBDIR "/conf/model.conf");
	if (!cfg) return nullptr;
	char *s = strstr(cfg, "/conf");
#ifdef _WIN32
	if (!s) s = strstr(cfg, "\\conf");
#endif
	if (s) *s = '\0';
	return cfg;
}

static cv::Mat frame_to_mat(struct obs_source_frame *frame)
{
	if (!frame || !frame->data[0]) return cv::Mat();
	uint32_t w = frame->width;
	uint32_t h = frame->height;
	if (frame->format == VIDEO_FORMAT_NV12) {
		cv::Mat y(h, (int)w, CV_8UC1, frame->data[0], frame->linesize[0]);
		cv::Mat uv(h / 2, (int)w / 2, CV_8UC2, frame->data[1], frame->linesize[1]);
		cv::Mat bgr;
		cv::cvtColorTwoPlane(y, uv, bgr, cv::COLOR_YUV2BGR_NV12);
		return bgr;
	}
	if (frame->format == VIDEO_FORMAT_I420) {
		cv::Mat y(h, (int)w, CV_8UC1, frame->data[0], frame->linesize[0]);
		cv::Mat u(h/2, (int)w/2, CV_8UC1, frame->data[1], frame->linesize[1]);
		cv::Mat v(h/2, (int)w/2, CV_8UC1, frame->data[2], frame->linesize[2]);
		cv::Mat bgr;
		cv::Mat u_resized, v_resized;
		cv::resize(u, u_resized, cv::Size((int)w, (int)h));
		cv::resize(v, v_resized, cv::Size((int)w, (int)h));
		cv::Mat yuv;
		cv::merge(std::vector<cv::Mat>{y, u_resized, v_resized}, yuv);
		cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);
		return bgr;
	}
	if (frame->format == VIDEO_FORMAT_YUY2) {
		cv::Mat yuy2((int)h, (int)w, CV_8UC2, frame->data[0], frame->linesize[0]);
		cv::Mat bgr;
		cv::cvtColor(yuy2, bgr, cv::COLOR_YUV2BGR_YUY2);
		return bgr;
	}
	if (frame->format == VIDEO_FORMAT_BGRA) {
		return cv::Mat((int)h, (int)w, CV_8UC4, frame->data[0], frame->linesize[0]).clone();
	}
	if (frame->format == VIDEO_FORMAT_BGRX) {
		cv::Mat bgrx((int)h, (int)w, CV_8UC4, frame->data[0], frame->linesize[0]);
		cv::Mat bgr;
		cv::cvtColor(bgrx, bgr, cv::COLOR_BGRA2BGR);
		return bgr;
	}
	return cv::Mat();
}

static void mat_to_frame(const cv::Mat &bgr, struct obs_source_frame *frame)
{
	if (!frame || !frame->data[0] || bgr.empty()) return;
	uint32_t w = frame->width;
	uint32_t h = frame->height;
	if (frame->format == VIDEO_FORMAT_BGRA || frame->format == VIDEO_FORMAT_BGRX) {
		cv::Mat out((int)h, (int)w, CV_8UC4, frame->data[0], frame->linesize[0]);
		if (bgr.channels() == 3)
			cv::cvtColor(bgr, out, cv::COLOR_BGR2BGRA);
		else
			bgr.copyTo(out);
		return;
	}
	if (frame->format == VIDEO_FORMAT_YUY2) {
		cv::Mat yuy2((int)h, (int)w, CV_8UC2, frame->data[0], frame->linesize[0]);
		cv::cvtColor(bgr, yuy2, cv::COLOR_BGR2YUV_YUY2);
	}
}

static double now_seconds()
{
	using namespace std::chrono;
	return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static const char *filter_name(void *) { return "Face Captions"; }

static void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	auto *gf = new face_captions_filter_data;
	gf->context = source;
	gf->face_detector = std::make_unique<FaceDetector>();
	gf->kalman = std::make_unique<KalmanTracker>();
	gf->caption_renderer = std::make_unique<CaptionRenderer>();
	obs_source_update(source, settings);
	return gf;
}

static void filter_destroy(void *data)
{
	auto *gf = static_cast<face_captions_filter_data *>(data);
	gf->active = false;
	gf->model_load_abort = true;
	if (gf->model_load_thread.joinable()) gf->model_load_thread.join();
	if (gf->stt) gf->stt->stop();
	delete gf;
}

static void filter_update(void *data, obs_data_t *settings)
{
	auto *gf = static_cast<face_captions_filter_data *>(data);
	gf->font_size = (int)obs_data_get_int(settings, S_FONT_SIZE);
	if (gf->font_size < 12) gf->font_size = 12;
	if (gf->font_size > 120) gf->font_size = 120;
	gf->caption_max_width = (int)obs_data_get_int(settings, S_MAX_WIDTH);
	if (gf->caption_max_width < 100) gf->caption_max_width = 100;
	gf->caption_offset_y = (int)obs_data_get_int(settings, S_OFFSET_Y);
	gf->show_face_box = obs_data_get_bool(settings, S_SHOW_FACE_BOX);
	gf->speech_bubble_tail = obs_data_get_bool(settings, S_SPEECH_BUBBLE);
	gf->caption_history_lines = (int)obs_data_get_int(settings, S_HISTORY_LINES);
	if (gf->caption_history_lines < 0) gf->caption_history_lines = 0;
	if (gf->caption_history_lines > 2) gf->caption_history_lines = 2;
	gf->chroma_key_mode = obs_data_get_bool(settings, S_CHROMA_MODE);
	const char *cc = obs_data_get_string(settings, S_CHROMA_COLOR);
	gf->chroma_color = cc ? cc : "green";
	int rate = (int)obs_data_get_int(settings, S_AUDIO_RATE);
	gf->input_audio_rate = (rate == 44100 || rate == 48000) ? rate : 44100;

	const char *model = obs_data_get_string(settings, S_VOSK_MODEL);
	std::string model_str = model ? model : "";
	if (model_str.empty()) {
#ifdef HAVE_VOSK
		char *best = get_best_vosk_model_path();
		if (best) { model_str = best; bfree(best); }
#endif
	}
	if (model_str != gf->vosk_model_path) {
#ifdef HAVE_VOSK
		gf->stt_ready = false;
		if (gf->model_load_thread.joinable()) { gf->model_load_abort = true; gf->model_load_thread.join(); gf->model_load_abort = false; }
		if (gf->stt) { gf->stt->stop(); gf->stt.reset(); }
#endif
		gf->vosk_model_path = model_str;
#ifdef HAVE_VOSK
		if (!model_str.empty()) {
			if (model_str.find("osk-model") == 0) model_str = "v" + model_str;
			bool path_ok = false;
			std::string conf_path = model_str;
			if (conf_path.back() != '/' && conf_path.back() != '\\') conf_path += "/";
			conf_path += "conf/model.conf";
			FILE *f = nullptr;
#ifdef _WIN32
			fopen_s(&f, conf_path.c_str(), "r");
#else
			f = fopen(conf_path.c_str(), "r");
#endif
			if (f) { fclose(f); path_ok = true; }
			if (!path_ok) {
				char *bundled = get_bundled_model_path();
				if (bundled) { model_str = bundled; bfree(bundled); }
			}
			gf->vosk_model_path = model_str;
			gf->stt = std::make_unique<STTVosk>();
			obs_source_t *parent = obs_filter_get_parent(gf->context);
			if (parent) {
				struct obs_audio_info oai;
				obs_get_audio_info(&oai);
				gf->obs_audio_rate = (int)oai.samples_per_sec;
				if (gf->obs_audio_rate <= 0) gf->obs_audio_rate = 48000;
			}
			std::string path = gf->vosk_model_path;
			face_captions_filter_data *gf_capture = gf;
			gf->model_load_abort = false;
			gf->model_load_thread = std::thread([gf_capture, path]() {
				if (gf_capture->model_load_abort || !gf_capture->stt) return;
				bool ok = gf_capture->stt->start(path.c_str(), 16000, [gf_capture](const char *text, bool is_final) {
					std::lock_guard<std::mutex> lock(gf_capture->text_mutex);
					double now = now_seconds();
					if (is_final) {
						std::string final_text = text ? text : "";
						if (!final_text.empty()) {
							gf_capture->caption_text = final_text;
							gf_capture->last_caption_time = now;
							if (gf_capture->caption_history_lines > 0) {
								gf_capture->caption_history[1] = gf_capture->caption_history[0];
								gf_capture->caption_history[0] = gf_capture->caption_text;
							}
						}
						gf_capture->caption_partial.clear();
					} else {
						std::string partial = text ? text : "";
						if (!partial.empty()) {
							gf_capture->caption_partial = partial;
							gf_capture->last_caption_time = now;
						}
					}
				});
				if (!ok) gf_capture->stt.reset();
				gf_capture->stt_ready = true;
			});
		}
#endif
	}
	gf->caption_renderer->set_font_size(gf->font_size);
	gf->caption_renderer->set_max_width(gf->caption_max_width);
	gf->caption_renderer->set_speech_bubble_tail(gf->speech_bubble_tail);
}

static struct obs_audio_data *filter_audio(void *data, struct obs_audio_data *audio)
{
	auto *gf = static_cast<face_captions_filter_data *>(data);
	if (gf->stt_ready && gf->stt && gf->stt->is_running() && audio && audio->data[0]) {
		size_t n = audio->frames;
		int in_sr = (gf->input_audio_rate == 44100 || gf->input_audio_rate == 48000) ? gf->input_audio_rate : 44100;
		int out_sr = 16000;
		size_t out_n = (n * (size_t)out_sr) / (size_t)in_sr;
		if (out_n > 0) {
			std::vector<int16_t> buf(out_n);
			const float *ch0 = (const float *)audio->data[0];
			const float *ch1 = audio->data[1] ? (const float *)audio->data[1] : nullptr;
			for (size_t i = 0; i < out_n; i++) {
				size_t start = (i * (size_t)in_sr) / (size_t)out_sr;
				size_t end = ((i + 1) * (size_t)in_sr) / (size_t)out_sr;
				if (end > n) end = n;
				if (start >= end) { buf[i] = 0; continue; }
				float sum = 0.f;
				for (size_t j = start; j < end; j++) {
					sum += ch0[j];
					if (ch1) sum += ch1[j];
				}
				sum /= (float)(ch1 ? (2 * (end - start)) : (end - start));
				float s = sum * 32767.f;
				buf[i] = (int16_t)(s < -32768.f ? -32768 : (s > 32767.f ? 32767 : (int16_t)s));
			}
			gf->stt->feed_audio(buf.data(), out_n * sizeof(int16_t));
		}
	}
	return audio;
}

static struct obs_source_frame *filter_video(void *data, struct obs_source_frame *frame)
{
	auto *gf = static_cast<face_captions_filter_data *>(data);
	if (!frame || !frame->data[0]) return frame;
	gf->active = true;

	cv::Mat bgr = frame_to_mat(frame);
	if (bgr.empty()) return frame;

	int w = bgr.cols, h = bgr.rows;
	gf->base_width = (uint32_t)w;
	gf->base_height = (uint32_t)h;
	gf->frame_count++;

	// --- Face detection with frame skipping and downscaling ---
	float cx = (gf->face_center_x > 0 && gf->face_center_x < (float)w) ? gf->face_center_x : w / 2.f;
	float ty = (gf->face_top_y > 0 && gf->face_top_y < (float)h) ? gf->face_top_y : h / 3.f;

	cv::Rect face = gf->face_detector->detect(bgr, cx, ty, gf->face_detected);
	if (face.width > 0 && face.height > 0) {
		gf->face_center_x = face.x + face.width * 0.5f;
		gf->face_top_y = (float)face.y;
		gf->face_bbox[0] = face.x; gf->face_bbox[1] = face.y;
		gf->face_bbox[2] = face.width; gf->face_bbox[3] = face.height;
		gf->face_detected = true;
		gf->kalman->update(gf->face_center_x, gf->face_top_y, face.width, face.height);
	} else {
		gf->face_detected = false;
		gf->kalman->predict(&gf->face_center_x, &gf->face_top_y);
	}

	// Apply velocity lookahead (like Python: cx += vx * lookahead_frames)
	float vx_k = 0.f, vy_k = 0.f;
	gf->kalman->get_velocity(&vx_k, &vy_k);
	const float lookahead = 2.0f;
	float smoothed_cx = gf->face_center_x + vx_k * lookahead;
	float smoothed_ty = gf->face_top_y + vy_k * lookahead;
	// Clamp to frame
	smoothed_cx = std::max(0.f, std::min(smoothed_cx, (float)w));
	smoothed_ty = std::max(0.f, std::min(smoothed_ty, (float)h));

	// --- Build display text ---
	std::string display_text;
	{
		std::lock_guard<std::mutex> lock(gf->text_mutex);
		double now = now_seconds();

		// Show partial (live typing) if available
		if (!gf->caption_partial.empty()) {
			display_text = gf->caption_partial;
		}
		// Otherwise show last final caption if not timed out
		else if (!gf->caption_text.empty()) {
			double elapsed = now - gf->last_caption_time;
			if (elapsed < gf->CAPTION_TIMEOUT) {
				// Build display with history
				std::string history_text;
				if (gf->caption_history_lines > 0 && !gf->caption_history[0].empty()) {
					if (gf->caption_history_lines > 1 && !gf->caption_history[1].empty()) {
						history_text = gf->caption_history[1] + "\n";
					}
					history_text += gf->caption_history[0];
				} else {
					history_text = gf->caption_text;
				}
				display_text = history_text;
			}
		}
		// Show status when there's no speech text (so user sees the plugin is active)
		if (display_text.empty()) {
#ifdef HAVE_VOSK
			if (!gf->stt && !gf->vosk_model_path.empty())
				display_text = "Loading model...";
			else if (gf->stt && !gf->stt_ready)
				display_text = "Loading model...";
			else if (gf->stt_ready && gf->stt && gf->stt->is_running())
				display_text = "Listening...";
			else if (!gf->stt)
				display_text = "Set Vosk model path in filter settings";
#else
			display_text = "Listening...";
#endif
		}
	}

	// --- Render and composite caption ---
	if (gf->caption_renderer && !display_text.empty()) {
		cv::Mat caption_rgba = gf->caption_renderer->render(display_text);
		if (!caption_rgba.empty()) {
			int cw = caption_rgba.cols, ch = caption_rgba.rows;
			// Position: above face, centered (like Python)
			int face_h = gf->face_bbox[3] > 0 ? gf->face_bbox[3] : 80;
			int dynamic_offset = std::max(55, (int)(face_h * 0.3f));
			float raw_capt_y = smoothed_ty - ch - dynamic_offset + gf->caption_offset_y;
			float raw_capt_x = smoothed_cx - cw / 2.f;

			// Smooth caption position (EMA)
			const float smooth_alpha = 0.82f;
			if (gf->smooth_capt_valid) {
				gf->smooth_capt_x = smooth_alpha * gf->smooth_capt_x + (1.f - smooth_alpha) * raw_capt_x;
				gf->smooth_capt_y = smooth_alpha * gf->smooth_capt_y + (1.f - smooth_alpha) * raw_capt_y;
			} else {
				gf->smooth_capt_x = raw_capt_x;
				gf->smooth_capt_y = raw_capt_y;
				gf->smooth_capt_valid = true;
			}

			int capt_x = (int)gf->smooth_capt_x;
			int capt_y = (int)gf->smooth_capt_y;
			// Clamp to frame bounds
			capt_x = std::max(0, std::min(capt_x, w - cw));
			capt_y = std::max(0, std::min(capt_y, h - ch));

			// Alpha composite (BGRA caption onto BGR frame)
			for (int y = 0; y < ch; y++) {
				int fy = capt_y + y;
				if (fy < 0 || fy >= h) continue;
				const cv::Vec4b *src_row = caption_rgba.ptr<cv::Vec4b>(y);
				for (int x = 0; x < cw; x++) {
					int fx = capt_x + x;
					if (fx < 0 || fx >= w) continue;
					const cv::Vec4b &p = src_row[x];
					if (p[3] == 0) continue;
					float a = p[3] / 255.f;
					float ia = 1.f - a;
					if (bgr.channels() == 3) {
						cv::Vec3b &dst = bgr.at<cv::Vec3b>(fy, fx);
						dst[0] = (uint8_t)(p[0] * a + dst[0] * ia);
						dst[1] = (uint8_t)(p[1] * a + dst[1] * ia);
						dst[2] = (uint8_t)(p[2] * a + dst[2] * ia);
					}
				}
			}
		}
	}

	// Draw face box if enabled (after caption so it's on top)
	if (gf->show_face_box && face.width > 0)
		cv::rectangle(bgr, face, cv::Scalar(0, 255, 0), 2);

	mat_to_frame(bgr, frame);
	return frame;
}

static obs_properties_t *filter_properties(void *data)
{
	(void)data;
	obs_properties_t *props = obs_properties_create();
	obs_properties_add_int(props, S_FONT_SIZE, "Caption Font Size", 12, 120, 1);
	obs_properties_add_int(props, S_MAX_WIDTH, "Caption Max Width", 200, 1000, 10);
	obs_properties_add_int(props, S_OFFSET_Y, "Caption Offset Y", -80, 40, 5);
	obs_properties_add_bool(props, S_SHOW_FACE_BOX, "Show Face Box");
	obs_properties_add_bool(props, S_SPEECH_BUBBLE, "Speech Bubble Tail");
	obs_properties_add_int(props, S_HISTORY_LINES, "Caption History Lines", 0, 2, 1);
	obs_properties_add_bool(props, S_CHROMA_MODE, "Chroma Key Mode");
	obs_properties_add_text(props, S_VOSK_MODEL, "Vosk Model Path", OBS_TEXT_DEFAULT);
	obs_property_t *rate_list = obs_properties_add_list(props, S_AUDIO_RATE, "Audio input sample rate (Hz)", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(rate_list, "44100 Hz", 44100);
	obs_property_list_add_int(rate_list, "48000 Hz", 48000);
	return props;
}

static void filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, S_FONT_SIZE, 52);
	obs_data_set_default_int(settings, S_MAX_WIDTH, 620);
	obs_data_set_default_int(settings, S_OFFSET_Y, 0);
	obs_data_set_default_bool(settings, S_SHOW_FACE_BOX, false);
	obs_data_set_default_bool(settings, S_SPEECH_BUBBLE, true);
	obs_data_set_default_int(settings, S_HISTORY_LINES, 2);
	obs_data_set_default_bool(settings, S_CHROMA_MODE, false);
	obs_data_set_default_int(settings, S_AUDIO_RATE, 44100);
}

obs_source_info face_captions_filter_info = {
	"face_captions_filter",
	OBS_SOURCE_TYPE_FILTER,
	OBS_SOURCE_VIDEO | OBS_SOURCE_AUDIO,
	filter_name,
	filter_create,
	filter_destroy,
	nullptr,
	nullptr,
	filter_defaults,
	filter_properties,
	filter_update,
	nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
	filter_video,
	filter_audio,
};
