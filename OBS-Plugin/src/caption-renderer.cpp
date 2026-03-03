#include "caption-renderer.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <sstream>

cv::Mat CaptionRenderer::render(const std::string &text)
{
	if (text.empty())
		return cv::Mat();

	// Check cache
	auto it = cache.find(text);
	if (it != cache.end())
		return it->second;

	int font = cv::FONT_HERSHEY_SIMPLEX;
	double font_scale = font_size / 30.0;
	int thickness = std::max(2, font_size / 18);
	int baseline = 0;

	// Word-wrap (matching Python _wrap_text)
	std::vector<std::string> lines;
	std::istringstream iss(text);
	std::string word, line;
	while (iss >> word) {
		std::string test = line.empty() ? word : line + " " + word;
		cv::Size sz = cv::getTextSize(test, font, font_scale, thickness, &baseline);
		if (sz.width > max_width && !line.empty()) {
			lines.push_back(line);
			line = word;
		} else {
			line = test;
		}
	}
	if (!line.empty())
		lines.push_back(line);
	// Keep only last 2 lines (like Python MAX_CAPTION_LINES = 2)
	if (lines.size() > 2)
		lines.erase(lines.begin(), lines.begin() + (int)(lines.size() - 2));

	// Layout
	int line_height = font_size + 6;
	int tail_h = speech_bubble_tail ? std::max(6, font_size * 10 / 52) : 0;
	int box_h = line_height * (int)lines.size() + padding * 2;
	int total_h = box_h + tail_h;

	int max_w = padding * 2;
	for (const auto &l : lines) {
		cv::Size sz = cv::getTextSize(l, font, font_scale, thickness, &baseline);
		max_w = std::max(max_w, sz.width + padding * 2);
	}
	int total_w = std::max(max_w, 100);

	cv::Mat img(total_h, total_w, CV_8UC4);
	img.setTo(cv::Scalar(0, 0, 0, 0));

	// Background fill
	cv::rectangle(img, cv::Rect(0, 0, total_w, box_h),
	              cv::Scalar(BG_B, BG_G, BG_R, BG_A), -1);

	// Border: Minecraft-style beveled edges (matching Python)
	// Light (white-ish) on top and left
	const int b = 2;
	for (int i = 0; i < b; i++) {
		// Top edge - light
		cv::line(img, cv::Point(i, i), cv::Point(total_w - 1 - i, i),
		         cv::Scalar(BORDER_LIGHT_B, BORDER_LIGHT_G, BORDER_LIGHT_R, 255), 1);
		// Left edge - light
		cv::line(img, cv::Point(i, i), cv::Point(i, box_h - 1 - i),
		         cv::Scalar(BORDER_LIGHT_B, BORDER_LIGHT_G, BORDER_LIGHT_R, 255), 1);
		// Bottom edge - dark
		cv::line(img, cv::Point(i, box_h - 1 - i), cv::Point(total_w - 1 - i, box_h - 1 - i),
		         cv::Scalar(BORDER_DARK_B, BORDER_DARK_G, BORDER_DARK_R, 255), 1);
		// Right edge - dark
		cv::line(img, cv::Point(total_w - 1 - i, i), cv::Point(total_w - 1 - i, box_h - 1 - i),
		         cv::Scalar(BORDER_DARK_B, BORDER_DARK_G, BORDER_DARK_R, 255), 1);
	}

	// Inner shadow (like Python: 1px darker line inside)
	int ofs = b;
	cv::line(img, cv::Point(ofs, ofs), cv::Point(total_w - 1 - ofs, ofs),
	         cv::Scalar(INNER_SHADOW_B, INNER_SHADOW_G, INNER_SHADOW_R, 255), 1);
	cv::line(img, cv::Point(ofs, ofs), cv::Point(ofs, box_h - 1 - ofs),
	         cv::Scalar(INNER_SHADOW_B, INNER_SHADOW_G, INNER_SHADOW_R, 255), 1);

	// Speech bubble tail
	if (speech_bubble_tail && tail_h > 0) {
		int cx = total_w / 2;
		int tail_w = 12;
		cv::Point pts[3] = {
			cv::Point(cx - tail_w, box_h - 1),
			cv::Point(cx + tail_w, box_h - 1),
			cv::Point(cx, total_h - 1),
		};
		cv::fillConvexPoly(img, pts, 3, cv::Scalar(BG_B, BG_G, BG_R, BG_A));
		// Dark outline on tail bottom
		cv::line(img, cv::Point(cx - tail_w, box_h - 1), cv::Point(cx, total_h - 1),
		         cv::Scalar(BORDER_DARK_B, BORDER_DARK_G, BORDER_DARK_R, 255), 1);
		cv::line(img, cv::Point(cx, total_h - 1), cv::Point(cx + tail_w, box_h - 1),
		         cv::Scalar(BORDER_DARK_B, BORDER_DARK_G, BORDER_DARK_R, 255), 1);
	}

	// Text rendering with shadow (matching Python shadow_offsets)
	int y = padding + baseline + font_size;
	for (const auto &l : lines) {
		const int pad = padding;
		// Shadow offsets: (2,2), (2,-2), (-2,2), (-2,-2) - like Python
		cv::putText(img, l, cv::Point(pad + 2, y + 2), font, font_scale, cv::Scalar(0, 0, 0, 200), thickness, cv::LINE_AA);
		cv::putText(img, l, cv::Point(pad + 2, y - 2), font, font_scale, cv::Scalar(0, 0, 0, 200), thickness, cv::LINE_AA);
		cv::putText(img, l, cv::Point(pad - 2, y + 2), font, font_scale, cv::Scalar(0, 0, 0, 200), thickness, cv::LINE_AA);
		cv::putText(img, l, cv::Point(pad - 2, y - 2), font, font_scale, cv::Scalar(0, 0, 0, 200), thickness, cv::LINE_AA);
		// Close shadow
		cv::putText(img, l, cv::Point(pad + 1, y + 1), font, font_scale, cv::Scalar(0, 0, 0, 255), thickness, cv::LINE_AA);
		// Main text - white
		cv::putText(img, l, cv::Point(pad, y), font, font_scale, cv::Scalar(255, 255, 255, 255), thickness, cv::LINE_AA);
		y += line_height;
	}

	// Store in cache (evict oldest if full)
	if (cache.size() >= MAX_CACHE) cache.clear();
	cache[text] = img;

	return img;
}
