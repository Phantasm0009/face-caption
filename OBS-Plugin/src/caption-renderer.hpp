#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <map>

struct CaptionRenderer {
	int font_size = 52;
	int max_width = 620;
	int padding = 14;
	bool speech_bubble_tail = true;

	// Minecraft-style colors (matching Python version)
	// BG: (45, 42, 34, 230)
	static constexpr int BG_R = 45, BG_G = 42, BG_B = 34, BG_A = 230;
	// Border light: top/left highlight
	static constexpr int BORDER_LIGHT_R = 90, BORDER_LIGHT_G = 85, BORDER_LIGHT_B = 72;
	// Border dark: bottom/right shadow
	static constexpr int BORDER_DARK_R = 20, BORDER_DARK_G = 18, BORDER_DARK_B = 15;
	// Inner shadow
	static constexpr int INNER_SHADOW_R = 30, INNER_SHADOW_G = 27, INNER_SHADOW_B = 19;

	// Simple caption cache (text -> rendered image)
	std::map<std::string, cv::Mat> cache;
	static constexpr size_t MAX_CACHE = 32;

	cv::Mat render(const std::string &text);
	void set_font_size(int s) { if (s != font_size) { font_size = s; cache.clear(); } }
	void set_max_width(int w) { if (w != max_width) { max_width = w; cache.clear(); } }
	void set_speech_bubble_tail(bool v) { if (v != speech_bubble_tail) { speech_bubble_tail = v; cache.clear(); } }
};
