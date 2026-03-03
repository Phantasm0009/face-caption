#include "stt-vosk.hpp"
#include <cstring>

#ifdef HAVE_VOSK
#include <vosk_api.h>
#include <queue>
#include <condition_variable>
#endif

struct STTVosk::Impl {
#ifdef HAVE_VOSK
	VoskModel *model = nullptr;
	VoskRecognizer *recognizer = nullptr;
	int sample_rate = 16000;
	std::function<void(const char *, bool)> callback;
	std::thread thread;
	std::mutex audio_mtx;
	std::queue<std::vector<char>> audio_queue;
	std::condition_variable audio_cv;
	std::atomic<bool> quit{false};
#endif
};

STTVosk::STTVosk() : impl(nullptr), running(false) {}

STTVosk::~STTVosk() { stop(); }

bool STTVosk::is_running() const { return running; }

bool STTVosk::start(const char *model_path, int sample_rate, std::function<void(const char *, bool)> callback)
{
#ifdef HAVE_VOSK
	if (!model_path || !model_path[0] || !callback) return false;
	if (impl) return false;
	impl = new Impl;
	impl->sample_rate = sample_rate;
	impl->callback = std::move(callback);
	impl->model = vosk_model_new(model_path);
	if (!impl->model) { delete impl; impl = nullptr; return false; }
	impl->recognizer = vosk_recognizer_new(impl->model, (float)sample_rate);
	if (!impl->recognizer) { vosk_model_free(impl->model); delete impl; impl = nullptr; return false; }
	impl->quit = false;
	running = true;
	impl->thread = std::thread(&STTVosk::worker_thread, this);
	return true;
#else
	(void)model_path; (void)sample_rate; (void)callback;
	return false;
#endif
}

void STTVosk::stop()
{
#ifdef HAVE_VOSK
	if (!impl) return;
	impl->quit = true;
	impl->audio_cv.notify_all();
	if (impl->thread.joinable()) impl->thread.join();
	if (impl->recognizer) vosk_recognizer_free(impl->recognizer);
	if (impl->model) vosk_model_free(impl->model);
	delete impl;
	impl = nullptr;
	running = false;
#endif
}

void STTVosk::feed_audio(const void *data, size_t bytes)
{
#ifdef HAVE_VOSK
	if (!impl || !data || bytes == 0) return;
	std::lock_guard<std::mutex> lock(impl->audio_mtx);
	impl->audio_queue.emplace((const char *)data, (const char *)data + bytes);
	impl->audio_cv.notify_one();
#else
	(void)data; (void)bytes;
#endif
}

void STTVosk::worker_thread()
{
#ifdef HAVE_VOSK
	if (!impl || !impl->recognizer) return;
	while (!impl->quit) {
		std::vector<char> chunk;
		{
			std::unique_lock<std::mutex> lock(impl->audio_mtx);
			impl->audio_cv.wait(lock, [this] { return impl->quit || !impl->audio_queue.empty(); });
			if (impl->quit) break;
			if (impl->audio_queue.empty()) continue;
			chunk = std::move(impl->audio_queue.front());
			impl->audio_queue.pop();
		}
		if (vosk_recognizer_accept_waveform(impl->recognizer, chunk.data(), (int)chunk.size())) {
			const char *result = vosk_recognizer_result(impl->recognizer);
			if (result && impl->callback) {
				const char *p = strstr(result, "\"text\"");
				if (p) {
					p = strchr(p + 7, '"');
					if (p) { p++; const char *end = strchr(p, '"'); if (end) impl->callback(std::string(p, end - p).c_str(), true); }
				}
			}
		} else {
			const char *partial = vosk_recognizer_partial_result(impl->recognizer);
			if (partial && impl->callback) {
				const char *p = strstr(partial, "\"partial\"");
				if (p) {
					p = strchr(p + 10, '"');
					if (p) { p++; const char *end = strchr(p, '"'); if (end) { std::string text(p, end - p); if (!text.empty()) impl->callback(text.c_str(), false); } }
				}
			}
		}
	}
#endif
}
