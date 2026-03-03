#pragma once

#include <functional>
#include <string>
#include <cstddef>

struct STTVosk {
	STTVosk();
	~STTVosk();

	bool start(const char *model_path, int sample_rate, std::function<void(const char *, bool)> callback);
	void stop();
	void feed_audio(const void *data, size_t bytes);
	bool is_running() const;

private:
	void worker_thread();
	struct Impl;
	Impl *impl;
	bool running;
};
