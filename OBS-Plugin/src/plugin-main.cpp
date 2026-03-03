#include <obs-module.h>
#include "face-captions-filter.hpp"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-face-captions", "en-US")

extern struct obs_source_info face_captions_filter_info;

bool obs_module_load(void)
{
	obs_register_source(&face_captions_filter_info);
	return true;
}

void obs_module_unload(void) {}

const char *obs_module_name(void)
{
	return "Face Captions";
}

const char *obs_module_description(void)
{
	return "Face-following captions with real-time speech-to-text. Add as a filter to your camera source.";
}
