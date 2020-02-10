/**
* This file is part of Intrinsic3D.
*
* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2019, Technical University of Munich. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*    * Redistributions of source code must retain the above copyright
*      notice, this list of conditions and the following disclaimer.
*    * Redistributions in binary form must reproduce the above copyright
*      notice, this list of conditions and the following disclaimer in the
*      documentation and/or other materials provided with the distribution.
*    * Neither the name of NVIDIA CORPORATION nor the names of its
*      contributors may be used to endorse or promote products derived
*      from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <nv/keyframe_selection.h>

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <iomanip>

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

namespace nv
{

    KeyframeSelection::KeyframeSelection(int window_size) :
        window_size_(window_size)
	{
	}


	KeyframeSelection::~KeyframeSelection()
	{
	}


	void KeyframeSelection::reset()
	{
		frame_scores_.clear();
		is_keyframe_.clear();
	}


	void KeyframeSelection::add(const cv::Mat &color)
	{
        // compute frame score (measure for image blur)
        double score = estimateBlur(color);

		frame_scores_.push_back(score);
	}


	void KeyframeSelection::selectKeyframes()
	{
		int n = (int)frame_scores_.size();

        // select best frame within a fixed size window
        is_keyframe_.resize(n, false);

        int num_windows = n / window_size_;
        if (n % window_size_ > 0)
            ++num_windows;

        for (int j = 0; j < num_windows; ++j)
        {
            // determine window start and end indices
            int id_win_beg = j * window_size_;
            int id_win_end = std::min(id_win_beg + window_size_, n);
            // find frame with best score within window
            double score_max = 0.0;
            int id_max = id_win_beg;
            for (int id = id_win_beg; id < id_win_end; ++id)
            {
                if (frame_scores_[id] > score_max)
                {
                    score_max = frame_scores_[id];
                    id_max = id;
                }
            }
            // assign isKeyframe to frame ids
            for (int id = id_win_beg; id < id_win_end; ++id)
            {
                is_keyframe_[id] = (id == id_max);
            }
        }
	}


	bool KeyframeSelection::isKeyframe(int id) const
	{
		if (!frameExists(id))
			return false;
		return is_keyframe_[id];
	}


    size_t KeyframeSelection::countKeyframes() const
    {
        size_t cnt = 0;
        for (auto is_kf : is_keyframe_)
        {
            if (is_kf)
                ++cnt;
        }
        return cnt;
    }


    void KeyframeSelection::drawScore(int id, cv::Mat &image) const
	{
		if (!frameExists(id))
			return;

        std::string text = "score: " + std::to_string(frame_scores_[id]);
        cv::putText(image, text, cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
	}


	bool KeyframeSelection::load(const std::string &filename)
    {
		reset();
		
		// open input file
		std::ifstream file(filename.c_str());
		if (!file.is_open())
			return false;

        // load window size
		std::string line;
		if (std::getline(file, line))
		{
			if (line.empty())
				return false;
            std::istringstream iss(line);
            int window_size;
            if (!(iss >> window_size))
                return false;
            window_size_ = window_size;
		}

		// load frames scores and isKeyframe from file
		while (std::getline(file, line))
		{
			if (line.empty())
				continue;
			std::istringstream iss(line);
			double score;
            bool is_kf;
            if (!(iss >> score >> is_kf))
				break;
			frame_scores_.push_back(score);
            is_keyframe_.push_back(is_kf);
		}

		// close file
		file.close();

		return true;
	}


	bool KeyframeSelection::save(const std::string &filename) const
	{
		if (frame_scores_.empty() || is_keyframe_.empty() || frame_scores_.size() != is_keyframe_.size())
			return false;

		// open output file
		std::ofstream file(filename.c_str());
		if (!file.is_open())
			return false;

        // write window size into output file
        file << window_size_ << std::endl;

		// write frame scores and isKeyframe into output file
		file << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < is_keyframe_.size(); ++i)
		{
			double score = frame_scores_[i];
			bool isKf = is_keyframe_[i];
            file << score << " " << static_cast<int>(isKf) << std::endl;
		}
		// close file
		file.close();

		return true;
	}


	bool KeyframeSelection::frameExists(int id) const
	{
        if (id < 0 || id >= frame_scores_.size() || id >= is_keyframe_.size())
			return false;
		else
			return true;
	}


	double KeyframeSelection::estimateBlur(const cv::Mat &color) const
	{
		if (color.empty())
			return 0.0;
		cv::Mat gray;
		if (color.channels() == 1)
			gray = color;
		else if (color.channels() == 3)
			cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
		else
			return 0.0;

		// convert grayscale image to float
        cv::Mat gray_f;
        gray.convertTo(gray_f, CV_32FC1, 1.0 / 255.0);

        // estimate blur using no-reference perceptual blur metric (Crete2007)
        return estimateBlurCrete(gray_f);
	}


	double KeyframeSelection::estimateBlurCrete(const cv::Mat &gray) const
	{
		// no-reference perceptual blur metric (Crete2007)

		int w = gray.cols;
		int h = gray.rows;

		// create blur kernels for strong low-pass filter
		cv::Mat hv = cv::Mat::ones(9, 1, CV_32F)  * (1.0 / 9.0);
		cv::Mat hh;
		cv::transpose(hv, hh);

		// blur image using a strong low-pass filter
        cv::Mat b_ver, b_hor;
        cv::filter2D(gray, b_ver, CV_32FC1, hv);
        cv::filter2D(gray, b_hor, CV_32FC1, hh);

		// compute absolute difference images (vertical)
        cv::Mat d_f_ver = cv::Mat::zeros(h, w, CV_32FC1);
        cv::Mat d_b_ver = cv::Mat::zeros(h, w, CV_32FC1);
		for (int y = 1; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
                d_f_ver.at<float>(y, x) = std::abs(gray.at<float>(y, x) - gray.at<float>(y - 1, x));
                d_b_ver.at<float>(y, x) = std::abs(b_ver.at<float>(y, x) - b_ver.at<float>(y - 1, x));
			}
		}

		// compute absolute difference images (horizontal)
        cv::Mat d_f_hor = cv::Mat::zeros(h, w, CV_32FC1);
        cv::Mat d_b_hor = cv::Mat::zeros(h, w, CV_32FC1);
		for (int y = 0; y < h; ++y)
		{
			for (int x = 1; x < w; ++x)
			{
                d_f_hor.at<float>(y, x) = std::abs(gray.at<float>(y, x) - gray.at<float>(y, x - 1));
                d_b_hor.at<float>(y, x) = std::abs(b_hor.at<float>(y, x) - b_hor.at<float>(y, x - 1));
			}
		}

		// compute vertical and horizontal variation
        cv::Mat v_ver = cv::Mat::zeros(h, w, CV_32FC1);
        cv::Mat v_hor = cv::Mat::zeros(h, w, CV_32FC1);
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
                v_ver.at<float>(y, x) = std::max(0.0f, d_f_ver.at<float>(y, x) - d_b_ver.at<float>(y, x));
                v_hor.at<float>(y, x) = std::max(0.0f, d_f_hor.at<float>(y, x) - d_b_hor.at<float>(y, x));
			}
		}

		// compute sum of coefficients
        double s_f_ver = cv::sum(d_f_ver).val[0];
        double s_v_ver = cv::sum(v_ver).val[0];
        double s_f_hor = cv::sum(d_f_hor).val[0];
        double s_v_hor = cv::sum(v_hor).val[0];

		// normalize results
        double b_f_ver = (s_f_ver - s_v_ver) / s_f_ver;
        double b_f_hor = (s_f_hor - s_v_hor) / s_f_hor;

		// compute final score: select more annyoing blur
        double blur_score = std::max(b_f_ver, b_f_hor);

		// invert score for 1.0=best and 0.0=worst
        blur_score = 1.0 - blur_score;

        return blur_score;
	}

} // namespace nv
